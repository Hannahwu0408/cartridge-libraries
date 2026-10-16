"""langgraph_agent.py — dynamic-depth ReAct agent over the cartridge library (the new-science arm).

THE QUESTION. Our chain controller is hard-coded 2-hop; the N-hop eval (eval_nhop.py) chains to a
KNOWN depth d. Real queries don't announce their depth — the deferred open item is a STOPPING
CRITERION. An agent loop supplies one for free: the policy LLM decides, step by step, whether to
search, consult a document, or finish. Here the SAME frozen Qwen3-4B is the policy; its tools are
the cartridge library (dense search over the frozen-text index; consult = answer from ONE entity's
cartridge at 100% weight — route-to-single as a tool). DEPTH IS HIDDEN from the agent.

Readouts (per true depth 2/3/4, n=30 each on data/synth_chain2):
  accuracy vs the scripted known-depth chain (run 2a: 0.60 / 0.47 / 0.40) vs one-shot;
  consults-used vs true depth (needs d+1: E0..Ed) → over-/under-hop rates = stopping behavior;
  JSON-action parse-failure rate (4B agentic-control health, target <5%); input tokens/query.
Either outcome is a finding: agent ≈ scripted ⇒ variable depth solved for free; agent < scripted ⇒
quantifies deterministic-controller vs 4B-agentic control (cf. the hop1-first confabulation result).

Actions (JSON, one per step — more robust than native tool-calling at 4B):
  {"action": "search",  "query": "<entity or topic>"}                → top-3 matching titles
  {"action": "consult", "title": "<document title>", "question": "<direct question>"}
  {"action": "finish",  "answer": "<final short answer>"}

Usage (one GPU):
  python scripts/langgraph_agent.py --index data/synth_chain2/index.json \
      --articles-file data/synth_chain2/corpus.jsonl \
      --multihop-file data/synth_chain2/multihop.jsonl \
      --doc-source cartridge --results-dir results/langgraph/agent_nhop
"""
from __future__ import annotations

import re
import sys
import json
import argparse
import itertools
from pathlib import Path
from typing import TypedDict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cartridge_runtime import CartridgeRuntime, load_jsonl, tok_list   # noqa: E402
from cartridge_qwen import generate_answer, load_model                 # noqa: E402
from evaluate_squad import containment, fuzzy_containment              # noqa: E402

try:
    from langgraph.graph import StateGraph, START, END
except ImportError:
    sys.exit("langgraph is not installed in this env — run:  pip install -U langgraph")

PROMPT = """You are answering a question using a library of documents. Each document describes ONE \
entity (a person, place, or work) and contains only facts about that entity.

Tools — reply with EXACTLY ONE JSON object and nothing else:
  {{"action": "search", "query": "<entity name or topic>"}}
      -> returns the best-matching document titles in the library.
  {{"action": "consult", "title": "<document title>", "question": "<a direct question>"}}
      -> answers your question using ONLY that one document.
  {{"action": "finish", "answer": "<the final short answer>"}}

{strategy}

Question: {question}

Steps so far:
{scratchpad}
Next action (one JSON object only):"""

# The strategy block is the ONLY thing --prompt-variant changes, so the comparison is controlled.
#
# `base` — what produced the recorded runs. Its two worked examples are 1-hop and 2-hop, which is a
#   confound for the depth result: an 8B policy stopped exactly one link short at every depth >= 3
#   while executing each hop perfectly, and we cannot tell miscounting from anchoring on these
#   examples without a variant that removes them.
# `counting` — no depth-suggestive examples; instead an explicit instruction to COUNT the repeated
#   relation and perform that many lookups. If the off-by-one disappears, the depth failure was
#   instruction-following, not an inability to track a chain.
STRATEGY = {
    "base":
        'Strategy: a question that refers to an entity indirectly (e.g. "the mentor of X", "the '
        'mentor of the mentor of X") must be resolved one link at a time: consult X\'s document to '
        "name X's mentor, then consult THAT person's document, and so on until you can answer. "
        "Never guess from memory — facts must come from consult results. When a consult result "
        "answers the original question, finish.",
    "counting":
        "Strategy: the question may reach its real subject through a CHAIN of indirect references. "
        "Before you act, COUNT how many times the linking relation is repeated in the question. "
        "That count is EXACTLY how many lookups you must perform before you may ask for the final "
        "attribute. Resolve them one at a time: consult the named entity's document to get the next "
        "entity, then consult THAT entity's document, and repeat. Keep track of how many lookups you "
        "have done so far; do NOT ask for the final attribute until you have completed all of them — "
        "stopping early gives a wrong answer. Never guess from memory: facts must come from consult "
        "results. Finish only once a consult result answers the original question.",
    # `recovery` — counting PLUS an explicit rule for the strongest correction signal the agent
    #   already receives and ignores: when it asks an intermediate entity for the final attribute,
    #   the reply is "the passage does not provide ...", which means "you have not gone far enough".
    #   Under `counting` the agent accepts that refusal as its answer (unforced) 93-100% of the time
    #   on deep chains. This variant is the LAST scaffold worth testing: each rule added here is a
    #   component the deterministic controller already implements, so if this is what it takes, the
    #   result is that the agent must be told to be the controller.
    "recovery":
        "Strategy: the question may reach its real subject through a CHAIN of indirect references. "
        "Before you act, COUNT how many times the linking relation is repeated in the question. "
        "That count is EXACTLY how many lookups you must perform before you may ask for the final "
        "attribute. Resolve them one at a time: consult the named entity's document to get the next "
        "entity, then consult THAT entity's document, and repeat.\n"
        "CRITICAL: if a consult replies that the document does not contain the attribute you asked "
        "for (e.g. 'the passage does not provide the town'), that means you have NOT yet reached the "
        "end of the chain — it is NOT the answer. Go back to the last entity you learned, consult its "
        "document for the NEXT entity in the chain, and continue. Never finish with a 'not provided' "
        "or 'not mentioned' reply; that is always a signal to keep going.\n"
        "Never guess from memory: facts must come from consult results. Finish only once a consult "
        "result actually states the attribute the question asked for.",
}

# ── round 2: three targeted scaffolds, each layered on `recovery` (the best prompt so far) so each
# reads as "what does this ADD". The failure has two separable parts:
#   (A) determining N — how many relation-hops the question requires
#   (B) tracking k    — how many are done, and continuing until k == N
#
# WHAT IS AND IS NOT PROVIDED (the line that keeps this agentic):
#   k — the agent's OWN completed-action count. The harness knows it from the trace; supplying it is
#       state management, which is precisely what an agent framework is for. NOT task knowledge.
#   N — NEVER provided. It must be derived from the question ("mentor of the mentor of ...").
#       Handing over N would disclose the hidden depth and reduce the agent to executing our plan,
#       i.e. the deterministic controller in disguise — the experiment would be meaningless.
#   In `all`, the annotation echoes "k of N planned" using the AGENT'S OWN stated N, never the truth.
#
# `plan` additionally MEASURES (A): the stated hop count is scored against the true depth, so we can
# tell "cannot count" from "cannot execute" — if the plan is right and it still stops early, the
# deficit is execution and `state` should repair it.
# UNIT BUGFIX (round-2 `state` cells were void). The plan asked for RELATION-LOOKUPS (= depth d) while
# the progress counter counted ALL consults (= d+1, since reading off the final attribute is also a
# consult). The agent therefore hit "3 of 3" after only 2 relation hops and stopped one link short —
# a mismatch in MY units, not an agent failure. Both now speak in TOTAL CONSULTS (d+1): one per link
# in the chain, plus one to read the answer. The counter already counts exactly that.
_PLAN_RULE = (
    "\nBEFORE any other action you MUST state how many documents you will need to consult IN TOTAL:\n"
    '  {{"action": "plan", "consults": <integer>}}\n'
    "Count one consult per link in the chain, PLUS one final consult to read off the answer. "
    'For example "In what town was the mentor of the mentor of X born?" needs 3 consults: X (to get '
    "the first mentor), that mentor (to get the second), and the second mentor (to read the town). "
    "Then execute exactly that many consults."
)

_STATE_NOTE = (
    "\nEach observation is annotated with how many documents you have consulted so far, counted the "
    "same way as the plan (every consult counts, including the final one that reads off the answer). "
    "Use it: while that count is below the total this question needs, keep consulting; you may read "
    "off the final attribute only on the LAST consult."
)

# Counted in TOTAL CONSULTS, matching _PLAN_RULE and the live progress annotation: 3 chain links plus
# one consult to read the answer = 4. (Round 2 numbered only the relation hops, which contradicted the
# counter whenever `state` was also on.)
_FEWSHOT = (
    '\n\nWorked example — "In what town was the mentor of the mentor of the mentor of Aldric Fenwick '
    'born?" Three links plus one consult to read the answer = 4 consults in total:\n'
    "  consult('Aldric Fenwick', 'Who is the mentor of Aldric Fenwick?')  -> Marta Kohl     [1 of 4]\n"
    "  consult('Marta Kohl', 'Who is the mentor of Marta Kohl?')          -> Devrin Oss     [2 of 4]\n"
    "  consult('Devrin Oss', 'Who is the mentor of Devrin Oss?')          -> Ilya Ravenna   [3 of 4]\n"
    "  consult('Ilya Ravenna', 'In what town was Ilya Ravenna born?')     -> Harrowgate     [4 of 4]\n"
    "  finish('Harrowgate')\n"
    "Note consult 3: asking Devrin Oss for a birthplace there would have been one link too early and "
    "would have returned 'not provided'. Only the LAST consult reads off the answer.\n"
    "Worked example — four links needs 5 consults in total:\n"
    "  consult(E0,'mentor?') -> E1 [1 of 5];  consult(E1,'mentor?') -> E2 [2 of 5];\n"
    "  consult(E2,'mentor?') -> E3 [3 of 5];  consult(E3,'mentor?') -> E4 [4 of 5];\n"
    "  consult(E4,'In what town was E4 born?') -> <town> [5 of 5];  finish(<town>)"
)

# Full 2^3 factorial over the three scaffolds, so any effect can be attributed rather than guessed.
# Canonical order (plan, state, fewshot) keeps each combination's text deterministic. Names join the
# active addons with "_": plan, state, fewshot, plan_state, plan_fewshot, state_fewshot,
# plan_state_fewshot (aliased "all"); the empty cell is `recovery`.
# `plan_state` is the mechanistically interesting pair: `state` alone shows "[lookups done: k]" with
# nothing to compare against, so N still lives only in the model's head; adding `plan` makes the
# harness able to display a COMPLETE progress state, "[lookups done: k of N planned]" — and N is
# still the agent's own number, never the truth.
_ADDON = {"plan": _PLAN_RULE, "state": _STATE_NOTE, "fewshot": _FEWSHOT}
_ORDER = ("plan", "state", "fewshot")
for _r in (1, 2, 3):
    for _combo in itertools.combinations(_ORDER, _r):
        STRATEGY["_".join(_combo)] = STRATEGY["recovery"] + "".join(_ADDON[c] for c in _combo)
STRATEGY["all"] = STRATEGY["plan_state_fewshot"]                       # convenience alias

# variants whose HARNESS behaviour differs (prompt text alone is not enough for these two)
NEEDS_PLAN_ACTION = {v for v in STRATEGY if "plan" in v or v == "all"}
NEEDS_PROGRESS_ANNOTATION = {v for v in STRATEGY if "state" in v or v == "all"}


class AgentState(TypedDict, total=False):
    question: str
    scratchpad: list
    n_steps: int
    n_consults: int
    n_parse_fail: int
    last_consult: str
    planned_hops: int
    action: dict
    prediction: str
    forced: bool
    route: str


def parse_action(text: str):
    """Extract the first JSON action object from a policy generation (single-quote repair included)."""
    m = re.search(r"\{.*?\}", text or "", re.DOTALL)
    if not m:
        return None
    for cand in (m.group(0), m.group(0).replace("'", '"')):
        try:
            d = json.loads(cand)
            if isinstance(d, dict) and d.get("action") in ("search", "consult", "finish", "plan"):
                return d
        except Exception:                                  # noqa: BLE001
            continue
    return None


def build_agent_app(policy, tool_search, tool_consult, max_steps: int, strategy: str = "base"):
    """The agent↔tool loop as a compiled LangGraph. Injected callables so it is testable with a
    scripted fake policy (no model) and reusable with the real runtime in main()."""
    strat = STRATEGY[strategy]
    annotate = strategy in NEEDS_PROGRESS_ANNOTATION

    def agent_node(s: AgentState):
        if s["n_steps"] >= max_steps:                      # budget exhausted → forced finish
            return {"prediction": s.get("last_consult", ""), "forced": True, "route": "end"}
        pad = "\n".join(s["scratchpad"]) if s["scratchpad"] else "(none yet)"
        act = parse_action(policy(PROMPT.format(strategy=strat, question=s["question"],
                                                scratchpad=pad)))
        if act is None:
            return {"n_parse_fail": s["n_parse_fail"] + 1, "n_steps": s["n_steps"] + 1,
                    "scratchpad": s["scratchpad"] +
                    [f"{s['n_steps']+1}. (your reply was not a single valid JSON action — retry)"],
                    "route": "agent"}
        if act.get("action") == "plan":
            # Record the agent's OWN stated hop count (first plan wins). Never compared against the
            # truth in-loop — only scored afterwards, so nothing about the real depth leaks back.
            #
            # BUGFIX (round-1 plan cells were void): the strategy text says "BEFORE any other action
            # you MUST state how many lookups...", and it is re-shown every turn, so after planning
            # the model simply re-planned until the step budget ran out — 10 plans, zero consults, on
            # every question. Once a plan exists we therefore refuse further plans and nudge the agent
            # to execute, rather than silently appending another plan line.
            if s.get("planned_hops") is not None:
                return {"n_steps": s["n_steps"] + 1, "route": "agent",
                        "scratchpad": s["scratchpad"] +
                        [f"{s['n_steps']+1}. (you have ALREADY planned {s['planned_hops']} lookups — "
                         f"do not plan again; take a search or consult action now)"]}
            raw = act.get("consults", act.get("hops"))   # "consults" is the current key; "hops" legacy
            try:
                hops = int(raw)
            except (TypeError, ValueError):
                hops = None
            upd = {"n_steps": s["n_steps"] + 1, "route": "agent",
                   "scratchpad": s["scratchpad"] +
                   [f"{s['n_steps']+1}. plan: this question needs {raw} consults in total "
                    f"— now execute them, do not plan again"]}
            if hops is not None:
                upd["planned_hops"] = hops
            return upd
        if act.get("action") == "finish":
            return {"action": act, "prediction": str(act.get("answer", "")),
                    "n_steps": s["n_steps"] + 1, "route": "end"}
        return {"action": act, "n_steps": s["n_steps"] + 1, "route": "tool"}

    def tool_node(s: AgentState):
        act = s["action"]
        if act["action"] == "search":
            obs = tool_search(str(act.get("query", "")))
            line = f"{s['n_steps']}. search({act.get('query', '')!r}) -> {obs}"
            return {"scratchpad": s["scratchpad"] + [line], "route": "agent"}
        obs = tool_consult(str(act.get("title", "")), str(act.get("question", "")))
        k = s["n_consults"] + 1
        # Progress annotation: the agent's own completed-lookup count, and — only if it stated one —
        # its own planned total. Never the true depth.
        tag = ""
        if annotate:
            plan = s.get("planned_hops")
            tag = f"   [lookups done: {k} of {plan} planned]" if plan else f"   [lookups done: {k}]"
        line = (f"{s['n_steps']}. consult({act.get('title', '')!r}, "
                f"{act.get('question', '')!r}) -> {obs}{tag}")
        return {"scratchpad": s["scratchpad"] + [line], "n_consults": k,
                "last_consult": obs, "route": "agent"}

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tool", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", lambda s: s["route"],
                            {"agent": "agent", "tool": "tool", "end": END})
    g.add_conditional_edges("tool", lambda s: s["route"], {"agent": "agent"})
    return g.compile()


def main():
    ap = argparse.ArgumentParser(description="Dynamic-depth ReAct agent over the cartridge library")
    ap.add_argument("--index", required=True)
    ap.add_argument("--articles-file", required=True)
    ap.add_argument("--multihop-file", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B",
                    help="the READER: answers consults. With --doc-source cartridge this MUST be the "
                         "cartridges' base model — a cartridge is trained KV in that model's "
                         "representation space and is meaningless to any other model.")
    ap.add_argument("--policy-model", default=None,
                    help="the POLICY: decides actions (default: same as --model-id). The policy never "
                         "touches a cartridge, so it is free to differ from the reader — decoupling "
                         "them isolates policy capability while holding the memory substrate fixed "
                         "(cf. CAS's decoupled question/answer models).")
    ap.add_argument("--policy-load-in-8bit", action="store_true",
                    help="load the policy in 8-bit so a larger policy can be co-resident with the 4B "
                         "reader on one 24 GB GPU (8B+4B in bf16 = 24.4 GB → OOM; 8-bit ≈ 16.5 GB). "
                         "~2-3x slower generation.")
    ap.add_argument("--doc-source", default="cartridge", choices=["cartridge", "text"],
                    help="what consult reads: the entity's cartridge KV, or its article text")
    ap.add_argument("--prompt-variant", default="base", choices=sorted(STRATEGY),
                    help="which strategy block to use. 'base' = the recorded runs (its examples are "
                         "1- and 2-hop, a confound for the depth result); 'counting' = no "
                         "depth-suggestive examples plus an explicit instruction to count the "
                         "repeated relation and do that many lookups. Only this block differs.")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--policy-max-tokens", type=int, default=96)
    ap.add_argument("--gen-max-tokens", type=int, default=48)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--results-dir", default="results/langgraph/agent")
    args = ap.parse_args()

    rt = CartridgeRuntime(args.index, model_id=args.model_id, gen_max_tokens=args.gen_max_tokens,
                          articles_file=args.articles_file)

    # Title → candidate slugs. NOTE the list: several documents can share a title (homonym twins in
    # the N-hop testbed; also real Wikipedia). Keeping only the first occurrence would hand the agent
    # an ORACLE disambiguator — in the generated corpus the real chain entity is always written
    # before its decoy, so first-occurrence == always correct, an advantage the scripted chain does
    # NOT have. Ambiguity is instead resolved by dense retrieval over the title, i.e. exactly the
    # coin flip the chain faces (see resolve_title).
    title2slugs: dict[str, list[str]] = {}
    for s in rt.ids:
        title2slugs.setdefault(rt.title_of_slug(s).lower(), []).append(s)
    n_ambig = sum(1 for v in title2slugs.values() if len(v) > 1)
    print(f"Index: {len(rt.ids)} documents, {n_ambig} titles shared by >1 document "
          f"(ambiguous → dense-resolved, no oracle)")

    # POLICY model. Defaults to the reader (one model, no extra memory). When a different policy is
    # named, BOTH are resident: the reader must stay the cartridges' base model, while the policy is
    # whatever we want to test. This is the only knob that isolates "is the agentic failure a policy
    # capability limit?" without retraining a single cartridge.
    if args.policy_model and args.policy_model != args.model_id:
        p_model, p_tok = load_model(args.policy_model, rt.device,
                                    load_in_8bit=args.policy_load_in_8bit)
        print(f"Decoupled policy: {args.policy_model}"
              f"{' [8-bit]' if args.policy_load_in_8bit else ''} | reader: {args.model_id}")
    else:
        p_model, p_tok = rt.model, rt.tokenizer

    counters = {"input_tokens": 0, "ambiguous_consults": 0,
                "doc_tokens": 0, "kv_positions": 0, "cart_reads": 0}

    def policy(prompt: str) -> str:
        counters["input_tokens"] += len(p_tok(prompt)["input_ids"])
        return generate_answer(p_model, p_tok, prompt, device=rt.device,
                               max_new_tokens=args.policy_max_tokens, conversational=True) or ""

    def resolve_title(title: str):
        """Agent-named title → slug. Exact title match decides ONLY when it is unambiguous; when
        several documents share the title the winner is chosen by dense similarity over the title —
        the same ambiguous retrieval the scripted chain performs — so the agent gets no oracle
        disambiguation. Falls back to a unique token-set match, then to plain dense retrieval."""
        t = (title or "").strip()
        cands = title2slugs.get(t.lower())
        if cands:
            if len(cands) == 1:
                return cands[0]
            counters["ambiguous_consults"] += 1              # homonym twins → dense tie-break
            v = rt.embed(t)
            sims = {s: float(rt.emb_mat[rt.ids.index(s)] @ v) for s in cands}
            return max(cands, key=lambda s: sims[s])
        tt = set(tok_list(t))
        if tt:
            hits = [s for s in rt.ids
                    if set(tok_list(re.sub(r"\(.*?\)", " ", rt.title_of_slug(s)))) == tt]
            if len(hits) == 1:
                return hits[0]
        got = rt.dense(rt.embed(t), 1)
        return got[0] if got else None

    def tool_search(query: str) -> str:
        sims = rt.emb_mat @ rt.embed(query)
        top = np.argsort(-sims)[:3]
        return "; ".join(f"{rt.title_of_slug(rt.ids[i])} ({sims[i]:.2f})" for i in top)

    def tool_consult(title: str, question: str) -> str:
        slug = resolve_title(title)
        if slug is None:
            return "(no such document)"
        counters["input_tokens"] += len(rt.tokenizer(question)["input_ids"])
        if args.doc_source == "text":
            # ACCOUNTING FIX: the article is prefilled alongside the question, so its tokens are
            # part of what the reader processes. Omitting them (as earlier runs did) undercounts
            # text mode and makes any cartridge-vs-text cost comparison meaningless.
            n = len(rt.tokenizer(rt.text_of([slug]))["input_ids"])
            counters["input_tokens"] += n
            counters["doc_tokens"] += n
            return rt.answer_text(question, [slug]) or "(no answer)"
        # cartridge mode: the document costs NO input tokens — it is already KV. Track the KV
        # positions it occupies instead, mirroring the §5.3 serving-cost convention.
        c = rt.load_cart(slug)
        if c is not None:
            counters["kv_positions"] += getattr(c, "cartridge_len", 128)
            counters["cart_reads"] += 1
        return rt.answer(question, [slug]) or "(no answer)"

    app = build_agent_app(policy, tool_search, tool_consult, args.max_steps,
                          strategy=args.prompt_variant)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        (out_dir / "graph.mmd").write_text(app.get_graph().draw_mermaid(), encoding="utf-8")
    except Exception as e:                                 # noqa: BLE001
        print(f"(mermaid export skipped: {e})")

    questions = load_jsonl(args.multihop_file)
    if args.max_questions:
        questions = questions[: args.max_questions]

    records = []
    for qi, q in enumerate(questions):
        t0, a0 = counters["input_tokens"], counters["ambiguous_consults"]
        d0, k0 = counters["doc_tokens"], counters["kv_positions"]
        out = app.invoke({"question": q["question"], "scratchpad": [], "n_steps": 0,
                          "n_consults": 0, "n_parse_fail": 0, "last_consult": "",
                          "planned_hops": None, "forced": False},
                         config={"recursion_limit": 6 * args.max_steps})
        pred = out.get("prediction", "")
        d = q.get("depth")
        rec = {"id": q.get("id", qi), "depth": d, "question": q["question"], "answer": q["answer"],
               "pred": pred, "fuzzy": fuzzy_containment(pred, q["answer"]),
               "cont": containment(pred, q["answer"]),
               "steps": out.get("n_steps", 0), "consults": out.get("n_consults", 0),
               "parse_fails": out.get("n_parse_fail", 0), "forced": out.get("forced", False),
               "input_tokens": counters["input_tokens"] - t0,
               "ambiguous_consults": counters["ambiguous_consults"] - a0,
               "planned_hops": out.get("planned_hops"),
               "doc_tokens": counters["doc_tokens"] - d0,
               "kv_positions": counters["kv_positions"] - k0,
               "scratchpad": out.get("scratchpad", [])}
        records.append(rec)
        print(f"[{qi+1}/{len(questions)}] {rec['id']} d={d} f={rec['fuzzy']:.0f} "
              f"steps={rec['steps']} consults={rec['consults']}"
              f"{' FORCED' if rec['forced'] else ''}{' PARSEFAIL' if rec['parse_fails'] else ''}")

    # ── per-depth report ────────────────────────────────────────────────────────
    depths = sorted({r["depth"] for r in records}, key=lambda x: (x is None, x))
    print("\n" + "=" * 78)
    pol = args.policy_model or args.model_id
    print(f"DYNAMIC-DEPTH AGENT — {len(records)} Qs · policy={pol}"
          f"{' [8bit]' if args.policy_load_in_8bit else ''} · reader={args.model_id} · "
          f"doc-source={args.doc_source} · prompt={args.prompt_variant} · "
          f"max-steps={args.max_steps}  (depth HIDDEN; needs d+1 consults)")
    print(f"{'depth':>5} {'n':>4} {'fuzzy':>6} {'consults μ':>10} {'=d+1':>6} {'>d+1':>6} "
          f"{'<d+1':>6} {'forced':>7} {'plan μ':>7} {'plan✓':>6} {'ambig/q':>8} {'tok/q':>7}")
    for d in depths:
        rs = [r for r in records if r["depth"] == d]
        n = len(rs)
        fz = sum(r["fuzzy"] for r in rs) / n
        cm = sum(r["consults"] for r in rs) / n
        if d is not None:
            eq = sum(r["consults"] == d + 1 for r in rs) / n
            over = sum(r["consults"] > d + 1 for r in rs) / n
            under = sum(r["consults"] < d + 1 for r in rs) / n
            eqs, ovs, uns = f"{eq:.2f}", f"{over:.2f}", f"{under:.2f}"
        else:
            eqs = ovs = uns = "—"
        forced = sum(r["forced"] for r in rs) / n
        pf = sum(r["parse_fails"] for r in rs) / max(1, sum(r["steps"] for r in rs))
        amb = sum(r.get("ambiguous_consults", 0) for r in rs) / n
        tok = sum(r["input_tokens"] for r in rs) / n
        # the agent's OWN stated plan, scored against the truth. The plan is stated in TOTAL CONSULTS,
        # so a correct plan for a depth-d chain is d+1 (d links plus one consult to read the answer).
        pl = [r["planned_hops"] for r in rs if r.get("planned_hops") is not None]
        plm = f"{sum(pl)/len(pl):>7.1f}" if pl else "      —"
        plok = (f"{sum(p == d + 1 for p in pl)/len(pl):>6.2f}"
                if (pl and d is not None) else "     —")
        print(f"{str(d):>5} {n:>4} {fz:>6.2f} {cm:>10.1f} {eqs:>6} {ovs:>6} {uns:>6} "
              f"{forced:>7.2f} {plm} {plok} {amb:>8.1f} {tok:>7.0f}")
    (out_dir / "agent_results.json").write_text(json.dumps(
        {"config": vars(args), "records": records}, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'agent_results.json'}")


if __name__ == "__main__":
    main()
