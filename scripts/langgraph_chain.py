"""langgraph_chain.py — the R1–R3 chain controller as a LangGraph StateGraph (parity arm).

WHAT THIS SHOWS. The cartridge library plugged into a standard agent framework as its document
memory: LangGraph orchestrates, each node calls our in-process frozen Qwen3-4B, and the "memory
read" is either a cartridge KV load or article text in context — switched by --doc-source with the
graph unchanged. (Named --doc-source, not --memory, to avoid colliding with LangGraph's own
checkpointer/store "memory" terminology.)

Two graph modes:
  monolithic — one node calls ChainController.run() directly: parity with router.py is exact BY
               CONSTRUCTION (same code path). The sanity anchor.
  nodes      — the controller decomposed into LangGraph nodes (retrieve_hop1 → decompose →
               validate_bridge → retrieve_hop2 → build_hop2_question → overhop_guard → answer),
               each calling the SAME ChainController step functions. The real integration + the
               paper figure. Verified per-question against monolithic / recorded router results.

⚠️ Port-correctness note: hop-2 RETRIEVAL is keyed on the decoded bridge-ENTITY string (or skipped
entirely via the R3 unique entity-link) — NEVER on the rewritten question. The rewrite is only the
answering prompt for the already-retrieved hop-2 source. Retrieve-then-rewrite is the code's order.

Usage (one GPU; needs `pip install langgraph`):
  python scripts/langgraph_chain.py --index cartridges/2wiki_v3_iso/index.json \
      --articles-file data/2wiki_v2/articles.jsonl --multihop-file data/2wiki_v2/multihop.jsonl \
      --doc-source cartridge --graph-mode nodes --results-dir results/langgraph/parity_cart \
      --compare-to results/chain_fix/R3/router_results.json
"""
from __future__ import annotations

import re
import sys
import json
import argparse
from pathlib import Path
from typing import TypedDict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cartridge_runtime as cr                                     # noqa: E402
from cartridge_runtime import (                                    # noqa: E402
    CartridgeRuntime, ChainController, load_jsonl, kinship_parse, tok_list,
    relation_leaks, is_refusal, RELWORDS, _AWARDISH,
)
from evaluate_squad import containment, fuzzy_containment, exact_match, f1_score  # noqa: E402

try:
    from langgraph.graph import StateGraph, START, END
except ImportError:
    sys.exit("langgraph is not installed in this env — run:  pip install -U langgraph")

# MUST stay byte-identical to router.SHORT / text_rag.SHORT — the matched-controller comparison is
# only meaningful if both substrates are decoded under the same instruction. Applied ONLY to the
# final answer (answer_node), never to bridge extraction, the reframe, or the overhop-guard probe,
# mirroring router.py's `sa()` placement.
SHORT = ("  Answer with ONLY the short answer itself (a few words, e.g. a name, place, or date) — "
         "no explanation, no full sentence.")


class ChainState(TypedDict, total=False):
    question: str
    q_emb: object            # np.ndarray (kept opaque; no checkpointer serialization needed)
    h1: list
    h1_slug: str
    h1_title: str
    kin: Optional[tuple]
    bridge: str
    attr: str
    h2: list
    answer_q: Optional[str]
    final_slugs: list
    prediction: str
    trace_bridge: str
    route: str


def build_node_graph(rt: CartridgeRuntime, ctrl: ChainController, short_answer: bool = False):
    """The controller decomposed into nodes. Every step calls the SAME ChainController methods
    router.py uses — the graph adds orchestration, not logic."""
    assert ctrl.template and ctrl.reframe, "node graph ports the deployed R1–R3 operating point"

    def retrieve_hop1(s: ChainState):
        q_emb = rt.embed(s["question"])
        h1 = rt.dense(q_emb, 1)
        return {"q_emb": q_emb, "h1": h1, "h1_slug": h1[0] if h1 else "",
                "h1_title": rt.title_of_slug(h1[0]) if h1 else "", "route": "decompose"}

    def decompose(s: ChainState):
        q, h1 = s["question"], s["h1"]
        kin = kinship_parse(q)
        if kin:
            phrase, parent, residual = kin
            # F2 — kinship over-hop: retrieval already landed on the parent article.
            ht_, qt_ = set(tok_list(s["h1_title"])), set(tok_list(q))
            if ht_ and len(ht_ & qt_) / len(ht_) < 0.6:
                return {"kin": kin, "final_slugs": h1, "trace_bridge": f"(kinship-redirect: {s['h1_title']})",
                        "answer_q": f"Who is the {residual} of {s['h1_title']}?", "route": "answer"}
            parent_q = re.sub(re.escape(phrase), parent, q, flags=re.IGNORECASE)
            raw = ctrl.answer(parent_q + " Reply with only the name.", h1) or ""
            bridge = (raw.strip().splitlines()[0].strip().strip('"').rstrip(".")
                      if raw.strip() else "")
            return {"kin": kin, "bridge": bridge, "attr": residual, "route": "validate_bridge"}
        bridge, attr = ctrl.bridge_and_attribute(q, h1)
        return {"kin": None, "bridge": bridge, "attr": attr, "route": "validate_bridge"}

    def validate_bridge(s: ChainState):
        if ctrl.bridge_valid(s["bridge"], s["h1_slug"], s["h1_title"]):
            return {"route": "retrieve_hop2"}
        note = (f"Note: the intermediate entity is NOT \"{s['h1_title']}\" and is not a bare role "
                f"word — name the OTHER specific person, organization, or work.")
        bridge, attr2 = ctrl.bridge_and_attribute(s["question"], s["h1"], extra_note=note)
        attr = s.get("attr") or attr2
        if ctrl.bridge_valid(bridge, s["h1_slug"], s["h1_title"]):
            return {"bridge": bridge, "attr": attr, "route": "retrieve_hop2"}
        return {"bridge": bridge, "attr": attr, "final_slugs": s["h1"], "route": "answer"}

    def retrieve_hop2(s: ChainState):
        # R3 unique entity-link → direct slug; else dense on the embedded BRIDGE-ENTITY string.
        slug_link, _hits = ctrl.link_bridge(s["bridge"])
        if slug_link and slug_link != s["h1_slug"]:
            h2 = [slug_link]
        else:
            h2 = rt.dense(rt.embed(s["bridge"]), 1, exclude=set(s["h1"]))
        if not h2:
            return {"h2": [], "final_slugs": s["h1"], "route": "answer"}
        return {"h2": h2, "route": "build_hop2_question"}

    def build_hop2_question(s: ChainState):
        q, bridge, attr = s["question"], s["bridge"], s.get("attr", "")
        if s.get("kin"):                                  # kinship: template IS the right question
            rq = f"Who is the {attr} of {bridge}?"
        else:
            work = ctrl.work_qualifier(q, s["h1_title"])
            rq = ctrl.reframe_question(q, bridge)
            if not ctrl.rewrite_ok(rq, q, bridge, attr, work):     # R3 referee: rewrite vs template
                at = set(tok_list(attr))
                if attr and (at & _AWARDISH) and work:
                    rq = f"Which {attr} did {bridge} win for {work}?"
                elif attr:
                    wh = "Who" if at & RELWORDS else "What"
                    rq = f"{wh} is the {attr} of {bridge}?"
                elif relation_leaks(rq, q, bridge):
                    for w in RELWORDS:
                        rq = re.sub(rf"(?i)\b{re.escape(bridge)}[’']s {w}\b", bridge, rq)
        nxt = "overhop_guard" if (ctrl.overhop_guard and s["h1_title"]) else "finalize_h2"
        return {"answer_q": rq, "route": nxt}

    def overhop_guard(s: ChainState):
        ht, qt = set(tok_list(s["h1_title"])), set(tok_list(s["question"]))
        if ht and len(ht & qt) / len(ht) < 0.6:           # subject-mismatch signature
            a2 = ctrl.answer(s["answer_q"], s["h2"])
            if is_refusal(a2):
                return {"final_slugs": s["h1"], "answer_q": None,
                        "trace_bridge": f"(overhop-guard: {s['bridge']})", "route": "answer"}
        return {"final_slugs": s["h2"], "route": "answer"}

    def finalize_h2(s: ChainState):
        return {"final_slugs": s["h2"], "route": "answer"}

    def answer_node(s: ChainState):
        q = s.get("answer_q") or s["question"]
        return {"prediction": ctrl.answer(q + SHORT if short_answer else q, s["final_slugs"])}

    g = StateGraph(ChainState)
    for name, fn in [("retrieve_hop1", retrieve_hop1), ("decompose", decompose),
                     ("validate_bridge", validate_bridge), ("retrieve_hop2", retrieve_hop2),
                     ("build_hop2_question", build_hop2_question),
                     ("overhop_guard", overhop_guard), ("finalize_h2", finalize_h2),
                     ("answer", answer_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve_hop1")
    route = lambda s: s["route"]
    g.add_conditional_edges("retrieve_hop1", route, {"decompose": "decompose"})
    g.add_conditional_edges("decompose", route,
                            {"validate_bridge": "validate_bridge", "answer": "answer"})
    g.add_conditional_edges("validate_bridge", route,
                            {"retrieve_hop2": "retrieve_hop2", "answer": "answer"})
    g.add_conditional_edges("retrieve_hop2", route,
                            {"build_hop2_question": "build_hop2_question", "answer": "answer"})
    g.add_conditional_edges("build_hop2_question", route,
                            {"overhop_guard": "overhop_guard", "finalize_h2": "finalize_h2"})
    g.add_conditional_edges("overhop_guard", route, {"answer": "answer"})
    g.add_conditional_edges("finalize_h2", route, {"answer": "answer"})
    g.add_edge("answer", END)
    return g.compile()


def build_monolithic_graph(ctrl: ChainController, short_answer: bool = False):
    """One node = ChainController.run(): parity with router.py exact by construction."""
    if short_answer:
        sys.exit("--short-answer requires --graph-mode nodes: the monolithic graph delegates the whole "
                 "chain to ChainController.run(), which owns its own final-answer call, so the terse "
                 "instruction cannot be applied to the final answer alone from out here.")

    def run_node(s: ChainState):
        pred, slugs = ctrl.run(s["question"])
        return {"prediction": pred, "final_slugs": slugs,
                "trace_bridge": ctrl.last_debug.get("bridge", ""),
                "answer_q": ctrl.last_debug.get("answer_q")}
    g = StateGraph(ChainState)
    g.add_node("chain_controller", run_node)
    g.add_edge(START, "chain_controller")
    g.add_edge("chain_controller", END)
    return g.compile()


def main():
    ap = argparse.ArgumentParser(description="LangGraph parity arm — cartridge library as agent doc memory")
    ap.add_argument("--index", required=True)
    ap.add_argument("--articles-file", required=True)
    ap.add_argument("--multihop-file", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--compose-mode", default="gnorm")
    ap.add_argument("--gen-max-tokens", type=int, default=48)
    ap.add_argument("--doc-source", default="cartridge", choices=["cartridge", "text"],
                    help="what a memory read returns: cartridge KV, or article text in context")
    ap.add_argument("--graph-mode", default="nodes", choices=["nodes", "monolithic"])
    ap.add_argument("--no-overhop-guard", action="store_true")
    ap.add_argument("--short-answer", action="store_true",
                    help="append the terse-span instruction to the FINAL answer only (byte-identical "
                         "to router.SHORT / text_rag.SHORT) so EM/token-F1 are comparable across "
                         "substrates; intermediates stay verbose. Requires --graph-mode nodes.")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--compare-to", default=None,
                    help="router_results.json to diff per-question chain preds against (parity check)")
    ap.add_argument("--results-dir", default="results/langgraph/parity")
    args = ap.parse_args()

    rt = CartridgeRuntime(args.index, model_id=args.model_id, compose_mode=args.compose_mode,
                          gen_max_tokens=args.gen_max_tokens, articles_file=args.articles_file)
    ctrl = ChainController(rt, template=True, overhop_guard=not args.no_overhop_guard,
                           reframe=True, doc_source=args.doc_source)

    # per-query input-token accounting (the cost half of the doc-source comparison)
    counters = {"calls": 0, "input_tokens": 0}
    orig_gen = rt.gen
    def counting_gen(**kw):
        counters["calls"] += 1
        text = (kw.get("question") or "") + " " + (kw.get("passage") or "")
        counters["input_tokens"] += len(rt.tokenizer(text)["input_ids"])
        return orig_gen(**kw)
    rt.gen = counting_gen                                  # instance attr shadows the method

    app = (build_node_graph(rt, ctrl, short_answer=args.short_answer) if args.graph_mode == "nodes"
           else build_monolithic_graph(ctrl, short_answer=args.short_answer))

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:                                                   # the paper figure
        (out_dir / "graph.mmd").write_text(app.get_graph().draw_mermaid(), encoding="utf-8")
    except Exception as e:                                 # noqa: BLE001 — figure is best-effort
        print(f"(mermaid export skipped: {e})")

    questions = load_jsonl(args.multihop_file)
    if args.max_questions:
        questions = questions[: args.max_questions]

    cmp_preds = {}
    if args.compare_to:
        ref = json.loads(Path(args.compare_to).read_text(encoding="utf-8"))
        cmp_preds = {r["id"]: r["policies"]["chain"]["pred"] for r in ref.get("records", [])
                     if "chain" in r.get("policies", {})}

    records, n_match = [], 0
    for qi, q in enumerate(questions):
        ctrl.last_debug.clear()
        c0, t0 = counters["calls"], counters["input_tokens"]
        out = app.invoke({"question": q["question"]})
        pred = out.get("prediction", "")
        rec = {"id": q.get("id", qi), "question": q["question"], "answer": q["answer"],
               "gold": q.get("article_slugs", []), "pred": pred,
               "final_slugs": out.get("final_slugs", []),
               "bridge": out.get("trace_bridge") or out.get("bridge", ""),
               "answer_q": out.get("answer_q"),
               "cont": containment(pred, q["answer"]), "fuzzy": fuzzy_containment(pred, q["answer"]),
               "em": exact_match(pred, q["answer"]), "f1": f1_score(pred, q["answer"]),
               "gen_calls": counters["calls"] - c0, "input_tokens": counters["input_tokens"] - t0}
        if rec["id"] in cmp_preds:
            rec["matches_reference"] = (pred == cmp_preds[rec["id"]])
            n_match += rec["matches_reference"]
        records.append(rec)
        print(f"[{qi+1}/{len(questions)}] {rec['id']} f={rec['fuzzy']:.0f} "
              f"tok={rec['input_tokens']}"
              + (f" ref={'=' if rec.get('matches_reference') else '≠'}" if cmp_preds else ""))

    n = len(records)
    fuzzy = sum(r["fuzzy"] for r in records) / n
    cont = sum(r["cont"] for r in records) / n
    em = sum(r["em"] for r in records) / n
    f1 = sum(r["f1"] for r in records) / n
    tok = sum(r["input_tokens"] for r in records) / n
    calls = sum(r["gen_calls"] for r in records) / n
    tag = "SHORT-ANSWER (EM/F1 comparable)" if args.short_answer else "verbose (fuzzy/cEM only)"
    print("\n" + "=" * 72)
    print(f"LANGGRAPH CHAIN — {n} Qs · graph={args.graph_mode} · doc-source={args.doc_source}")
    print(f"decoding: {tag}")
    print(f"fuzzy {fuzzy:.3f} · exact {cont:.3f} · EM {em:.3f} · F1 {f1:.3f} · "
          f"input-tokens/query {tok:.0f} · gen-calls/query {calls:.1f}")
    if cmp_preds:
        n_ref = sum(1 for r in records if "matches_reference" in r)
        print(f"parity vs {args.compare_to}: {n_match}/{n_ref} identical predictions")
    (out_dir / "langgraph_results.json").write_text(json.dumps(
        {"config": vars(args), "fuzzy": fuzzy, "exact": cont, "em": em, "f1": f1,
         "short_answer": args.short_answer, "input_tokens_per_q": tok,
         "gen_calls_per_q": calls, "records": records}, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'langgraph_results.json'}")


if __name__ == "__main__":
    main()
