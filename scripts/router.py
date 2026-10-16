"""
router.py — retrieve-and-compose router over a cartridge index (Phase 8, Stage 1).

Given the cartridge index (build_cartridge_index.py) and multihop queries with KNOWN gold
cartridges (`article_slugs`), evaluate retrieval policies and read the Stage-1 gates:

  all         — compose EVERY cartridge (interference baseline; expected to drown the answer)
  dense       — compose top-k by dense-kNN cosine similarity (sweeps every --top-k value)
  topic       — GMM over section embeddings → keep the query's top-m topic regions →
                dense-rerank within that subset → top-k  (needs sklearn; meaningful only when
                the library HAS topic structure — e.g. 2Wiki, not the synth crafts)
  iter        — ITERATIVE 2-hop retrieval for bridge questions: hop-1 retrieve top-k1 by the
                query and generate an intermediate answer (surfaces the bridge entity), then
                hop-2 re-embed (query + intermediate) and retrieve top-k2, compose the union.
                Attacks the failure mode where the 2nd gold is only reachable via the bridge
                entity the raw query never names — WITHOUT paying the interference cost of a
                larger k.

Reported per policy: retrieval recall of the gold cartridges, composition quality
(containment / fuzzy vs the gold answer), plus a teacher ceiling (gold articles in context).

  Gate A — does routing (best dense) beat compose-all?   (interference is the bottleneck)
  Gate B — does topic beat dense on quality?             (expected no at small scale)
  top-k tradeoff — recall rises with k, composition falls with k → sweet spot
  iter vs dense — does entity-augmented 2-hop retrieval lift recall without interference?

Usage
-----
python scripts/router.py \
    --index cartridges/synth_ca_gnorm/index.json \
    --multihop-file data/synth/multihop.jsonl \
    --articles-file data/synth/articles_mm.jsonl \
    --model-id Qwen/Qwen3-4B --cartridge-len 64 --compose-mode gnorm \
    --modes all,dense,iter --top-k 2,3,4 --results-dir results/router_synth
"""
from __future__ import annotations

import re
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cartridge_qwen import get_device  # noqa: E402
from cartridge_runtime import CartridgeRuntime, ChainController  # noqa: E402
from evaluate_squad import containment, exact_match, f1_score, fuzzy_containment  # noqa: E402

# MUST stay byte-identical to text_rag.SHORT — the cartridge chain and the text baselines have to
# be decoded under the same instruction or the EM/F1 comparison between them is meaningless.
SHORT = ("  Answer with ONLY the short answer itself (a few words, e.g. a name, place, or date) — "
         "no explanation, no full sentence.")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def cosine_topk(q: np.ndarray, mat: np.ndarray, k: int) -> list[int]:
    """Indices of the k rows of `mat` most cosine-similar to q (both L2-normalized upstream)."""
    sims = mat @ q
    return list(np.argsort(-sims)[:k])


def dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser(description="Retrieve-and-compose router (Stage 1)")
    ap.add_argument("--index", required=True, help="index.json from build_cartridge_index.py")
    ap.add_argument("--multihop-file", required=True,
                    help="queries with gold article_slugs (data/synth/multihop.jsonl)")
    ap.add_argument("--articles-file", required=True,
                    help="source sections, for the teacher ceiling (context in prompt)")
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--cartridge-len", type=int, default=64)
    ap.add_argument("--compose-mode", default="gnorm", choices=["gnorm", "concat"],
                    help="composition mechanism; gnorm = order-invariant (router can't order)")
    ap.add_argument("--gate", default=None,
                    help="trained RelevanceGate (gnorm only): learned per-cartridge weighting that "
                         "down-weights the diluting cartridge when one suffices and keeps both up "
                         "for synthesis (Phase 11). Expected to close the 2Wiki dilution tax.")
    ap.add_argument("--modes", default="all,dense",
                    help="retrieval policies to run, comma-sep: all,dense,topic,iter,fuse,chain "
                         "(fuse = RRF of query + bridge-entity rankings; chain = iterative "
                         "single-cartridge answering, no composition)")
    ap.add_argument("--top-k", default="2",
                    help="dense/topic top-k; comma-sep sweeps each value (e.g. '2,3,4')")
    ap.add_argument("--n-topics", type=int, default=8, help="GMM components (topic mode)")
    ap.add_argument("--top-m-topics", type=int, default=2, help="query topic regions to keep")
    ap.add_argument("--iter-k1", type=int, default=1, help="iter: cartridges retrieved at hop-1")
    ap.add_argument("--iter-k2", type=int, default=1, help="iter: cartridges added at hop-2")
    ap.add_argument("--fuse-bridge-weight", type=float, default=0.5,
                    help="fuse: weight on the bridge-entity ranking relative to the query ranking "
                         "(1.0 = equal; lower trusts the query more when bridge extraction is noisy)")
    ap.add_argument("--no-reframe", action="store_true",
                    help="chain: DISABLE hop-2 question reframing (ablation). By default chain rewrites "
                         "the multi-hop question into a direct question about the bridge entity so the "
                         "hop-2 cartridge answers from its OWN content (exploits strong direct fidelity)")
    ap.add_argument("--chain-template", action="store_true",
                    help="chain Fix A: extract (bridge, attribute) in one hop-1 call and build hop-2's "
                         "question as a clean '<attribute> of <bridge>' template when the free rewrite "
                         "still leaks the original relation word (the dominant chain failure).")
    ap.add_argument("--chain-hop1-first", action="store_true",
                    help="chain Fix C: answer directly from the hop-1 cartridge first; if it answers "
                         "without refusing, keep it. KNOWN-BAD (cartridges confabulate, don't refuse "
                         "— measured 0.45 vs 0.62); kept only for ablation.")
    ap.add_argument("--chain-overhop-guard", action="store_true",
                    help="chain R4: when hop-1's article is not the question's subject (retriever "
                         "already jumped the hop) and hop-2 refuses, answer the original question on "
                         "hop-1 instead — the narrow, refusal-gated version of hop1-first.")
    ap.add_argument("--iter-mode", default="bridge", choices=["bridge", "answer"],
                    help="iter hop-1 intermediate: 'bridge' extracts ONLY the next-lookup entity "
                         "(clean, avoids answering the final question early); 'answer' generates a "
                         "full answer to the query (the naive version — leaks/hallucinates)")
    ap.add_argument("--short-answer", action="store_true",
                    help="append a terse-span instruction at the ANSWERING step so strict-EM and "
                         "token-F1 are fair (verbose 'X was born in Y.' tanks F1 precision). "
                         "Required for numbers comparable to published EM/F1; contains-EM and "
                         "fuzzy are verbosity-robust either way.")
    ap.add_argument("--gen-max-tokens", type=int, default=48)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--no-teacher", action="store_true", help="skip the teacher ceiling")
    ap.add_argument("--show", type=int, default=0,
                    help="print readable per-query breakdowns for the first N queries "
                         "(-1 = all): question, gold, retrieved titles, per-cartridge single "
                         "preds, composed pred — to SEE what the cartridges are answering")
    ap.add_argument("--diagnostic", action="store_true",
                    help="for dense policies, also generate with each retrieved cartridge SINGLY, "
                         "so we can tell composition-dilution (compose >> best-single) from "
                         "single-cartridge fidelity / reasoning limits (both low)")
    ap.add_argument("--results-dir", default="results/router")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    klist = [int(k) for k in str(args.top_k).split(",") if str(k).strip()]
    device = get_device()
    print(f"Device: {device}")

    # ── runtime (index + frozen embedder + frozen model + cartridge cache) ───────────
    # Shared with the LangGraph integration — single source of truth in cartridge_runtime.py.
    rt = CartridgeRuntime(args.index, model_id=args.model_id, compose_mode=args.compose_mode,
                          gate_path=args.gate, gen_max_tokens=args.gen_max_tokens,
                          articles_file=args.articles_file, device=device)
    ids, id2entry, emb_mat = rt.ids, rt.id2entry, rt.emb_mat
    embed, dense, rank_ids = rt.embed, rt.dense, rt.rank_ids
    load_cart, answer, gen, articles = rt.load_cart, rt.answer, rt.gen, rt.articles
    sa = (lambda q: q + SHORT) if args.short_answer else (lambda q: q)
    print(f"Index: {len(ids)} cartridges")
    if rt.q_instr:
        print(f"Query instruction: {rt.q_instr!r}")
    if args.gate:
        print(f"RelevanceGate: {args.gate} (compose-mode must be gnorm to use it)")

    questions = load_jsonl(Path(args.multihop_file))
    if args.max_questions:
        questions = questions[: args.max_questions]

    # ── topic structure (only if requested) ────────────────────────────────────────
    gmm = None
    cart_topic = None
    if "topic" in modes:
        from sklearn.mixture import GaussianMixture
        K = min(args.n_topics, len(ids))
        gmm = GaussianMixture(n_components=K, covariance_type="full",
                              random_state=args.seed).fit(emb_mat)
        cart_topic = gmm.predict_proba(emb_mat).argmax(axis=1)  # each cartridge's hard topic
        print(f"Topic mode: fit GMM with {K} components over {len(ids)} embeddings")

    def topic(q_emb, k):
        q_topic = gmm.predict_proba(q_emb[None, :])[0]
        keep = set(np.argsort(-q_topic)[: args.top_m_topics])
        cand = [i for i in range(len(ids)) if cart_topic[i] in keep] or list(range(len(ids)))
        order = cosine_topk(q_emb, emb_mat[cand], min(k, len(cand)))
        return [ids[cand[i]] for i in order]

    # ── chain controller (R1–R4, F1–F3) — shared implementation in cartridge_runtime.py ──
    ctrl = ChainController(rt, template=args.chain_template,
                           overhop_guard=args.chain_overhop_guard,
                           reframe=not args.no_reframe, hop1_first=args.chain_hop1_first)
    last_debug = ctrl.last_debug          # per-query trace (bridge / rewritten answer_q)
    bridge_extract = ctrl.bridge_extract  # also used by the iter and fuse policies below
    chain = ctrl.chain

    def iterative(q_emb, q_text):
        """2-hop retrieval for bridge questions.
        hop-1: retrieve top-k1 by the query.
        hop-2: extract the bridge entity from hop-1 → embed IT → retrieve top-k2 among
               cartridges NOT already picked (so hop-2 finds the bridge-TARGET article)."""
        h1 = dense(q_emb, args.iter_k1)
        if args.iter_mode == "bridge":
            bridge = bridge_extract(q_text, h1)
            last_debug["bridge"] = bridge
            q2 = embed(bridge) if bridge.strip() else q_emb   # entity → closest to target article
        else:
            a1 = answer(q_text, h1)                           # naive: answer the whole question
            last_debug["bridge"] = a1
            q2 = embed(f"{q_text} {a1}")
        h2 = dense(q2, args.iter_k2, exclude=set(h1))         # force a NEW cartridge at hop-2
        return dedup(h1 + h2)

    def fuse(q_emb, q_text, k, k0=60):
        """Reciprocal-rank fusion of the query ranking and the bridge-entity ranking → top-k.
        Keeps dense's easy wins (B high in the query ranking) AND recovers the hard bridge
        cases (B high in the bridge ranking), without exceeding k (no interference cost)."""
        bridge = bridge_extract(q_text, dense(q_emb, args.iter_k1))
        last_debug["bridge"] = bridge
        b_emb = embed(bridge) if bridge.strip() else q_emb
        rq, rb = rank_ids(q_emb), rank_ids(b_emb)
        w = args.fuse_bridge_weight
        score = {s: 0.0 for s in ids}
        for r, s in enumerate(rq):
            score[s] += 1.0 / (k0 + r)
        for r, s in enumerate(rb):
            score[s] += w / (k0 + r)
        return sorted(ids, key=lambda s: -score[s])[:k]

    # Expand modes → concrete labeled policies (dense sweeps each k). Each policy maps a
    # query to its retrieved slug list; iter also needs the query text for its hop-1 answer.
    policies = []  # (label, fn(q_emb, q_text) -> slugs)
    for mode in modes:
        if mode == "all":
            policies.append(("all", lambda qe, qt: list(ids)))
        elif mode == "dense":
            for k in klist:
                policies.append((f"dense@{k}", (lambda k: lambda qe, qt: dense(qe, k))(k)))
        elif mode == "topic":
            for k in klist:
                policies.append((f"topic@{k}", (lambda k: lambda qe, qt: topic(qe, k))(k)))
        elif mode == "iter":
            policies.append((f"iter-{args.iter_mode}({args.iter_k1}+{args.iter_k2})",
                             lambda qe, qt: iterative(qe, qt)))
        elif mode == "fuse":
            for k in klist:
                policies.append((f"fuse@{k}", (lambda k: lambda qe, qt: fuse(qe, qt, k))(k)))
        elif mode == "chain":
            policies.append(("chain", lambda qe, qt: chain(qe, qt)))
        else:
            raise ValueError(f"unknown mode {mode}")
    labels = [lab for lab, _ in policies]

    # ── run ──────────────────────────────────────────────────────────────────────
    records = []
    for qi, q in enumerate(questions):
        q_text, gold_ans = q["question"], q["answer"]
        gold = set(q.get("article_slugs", []))
        q_emb = embed(q_text)

        rec = {"id": q.get("id", qi), "question": q_text, "answer": gold_ans,
               "gold": sorted(gold), "policies": {}}
        for lab, fn in policies:
            last_debug.clear()
            retr = fn(q_emb, q_text)
            # terse instruction only at the ANSWERING step — the intermediate bridge-extraction
            # prompts already ask for a bare name, and changing them would alter retrieval.
            pred = answer(sa(last_debug.get("answer_q", q_text)), retr)  # reframed Q if policy set one
            recall = (len(gold & set(retr)) / len(gold)) if gold else None
            rec["policies"][lab] = {
                "retrieved": retr, "n": len(retr), "recall": recall, "pred": pred,
                "cont": containment(pred, gold_ans), "fuzzy": fuzzy_containment(pred, gold_ans),
                "em": exact_match(pred, gold_ans), "f1": f1_score(pred, gold_ans),
                **({"bridge": last_debug["bridge"]} if "bridge" in last_debug else {}),
                **({"answer_q": last_debug["answer_q"]} if "answer_q" in last_debug else {}),
            }
            if args.diagnostic and lab.startswith("dense"):
                singles = []
                for s in retr:
                    c = load_cart(s)
                    if c is not None:
                        sp = gen(question=sa(q_text), cartridge=c)  # one cartridge alone
                        singles.append({"slug": s, "pred": sp,
                                        "fuzzy": fuzzy_containment(sp, gold_ans)})
                rec["policies"][lab]["singles"] = singles
                rec["policies"][lab]["single_best"] = max((x["fuzzy"] for x in singles), default=0.0)
        if not args.no_teacher:
            passage = "\n\n".join(articles[s]["context"] for s in q["article_slugs"] if s in articles)
            try:
                tp = gen(question=sa(q_text), passage=passage)
                rec["teacher"] = {"pred": tp, "cont": containment(tp, gold_ans),
                                  "fuzzy": fuzzy_containment(tp, gold_ans),
                                  "em": exact_match(tp, gold_ans),
                                  "f1": f1_score(tp, gold_ans)}
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  (teacher OOM on {rec['id']} — skipped; full-context prefill too long "
                      f"for this GPU)")
        records.append(rec)
        if device == "cuda":
            torch.cuda.empty_cache()
        line = f"[{qi+1}/{len(questions)}] {rec['id']}  " + "  ".join(
            f"{lab}:f={rec['policies'][lab]['fuzzy']:.0f}/r={rec['policies'][lab]['recall'] or 0:.1f}"
            for lab in labels)
        if not args.no_teacher:
            line += f"  T:f={rec['teacher']['fuzzy']:.0f}"
        print(line)

    # ── aggregate + gates ─────────────────────────────────────────────────────────
    n = len(records)
    def mean(lab, key):
        vals = [r["policies"][lab][key] for r in records if r["policies"][lab][key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n" + "=" * 72)
    print(f"ROUTER — {n} queries · compose={args.compose_mode}")
    tag = 'SHORT-ANSWER (EM/F1 comparable)' if args.short_answer else 'verbose (fuzzy/cEM only)'
    print(f"decoding: {tag}")
    print(f"{'policy':<14} {'recall':>7} {'cEM':>7} {'fuzzy':>7} {'EM':>7} {'F1':>7} "
          f"{'avg_n':>7}  {'compose|retr✓':>13}")
    summary = {}
    for lab in labels:
        avg_n = sum(r["policies"][lab]["n"] for r in records) / n
        # composition quality conditioned on full retrieval (recall==1) — isolates the
        # composition factor from the retrieval factor.
        full = [r for r in records if r["policies"][lab]["recall"] == 1.0]
        comp_given = (sum(r["policies"][lab]["fuzzy"] for r in full) / len(full)) if full else None
        summary[lab] = {"recall": mean(lab, "recall"), "exact": mean(lab, "cont"),
                        "fuzzy": mean(lab, "fuzzy"), "em": mean(lab, "em"),
                        "f1": mean(lab, "f1"), "avg_n": avg_n,
                        "compose_given_retrieval": comp_given, "n_full": len(full)}
        cg = f"{comp_given:.2f}({len(full)})" if comp_given is not None else "—"
        print(f"{lab:<14} {summary[lab]['recall']:>7.2f} {summary[lab]['exact']:>7.2f} "
              f"{summary[lab]['fuzzy']:>7.2f} {summary[lab]['em']:>7.2f} "
              f"{summary[lab]['f1']:>7.2f} {avg_n:>7.1f}  {cg:>13}")
    trecs = [r for r in records if "teacher" in r]
    if not args.no_teacher and trecs:
        tm = lambda k: sum(r["teacher"][k] for r in trecs) / len(trecs)
        tex, tfz, tem, tf1 = tm("cont"), tm("fuzzy"), tm("em"), tm("f1")
        summary["teacher"] = {"exact": tex, "fuzzy": tfz, "em": tem, "f1": tf1,
                              "n": len(trecs)}
        print(f"{'teacher':<14} {'—':>7} {tex:>7.2f} {tfz:>7.2f} {tem:>7.2f} {tf1:>7.2f} "
              f"{len(trecs):>7} (of {n})")

    print("-" * 72)
    dense_labs = [f"dense@{k}" for k in klist if f"dense@{k}" in summary]
    if "all" in summary and dense_labs:
        best = max(dense_labs, key=lambda l: summary[l]["fuzzy"])
        d = summary[best]["fuzzy"] - summary["all"]["fuzzy"]
        print(f"Gate A (routing > compose-all): {best} {summary[best]['fuzzy']:.2f} "
              f"vs all {summary['all']['fuzzy']:.2f}  Δ={d:+.2f}  → {'PASS' if d > 0 else 'FAIL'}")
    if len(dense_labs) > 1:
        print("top-k tradeoff (recall↑ vs composition↓):")
        for l in dense_labs:
            cg = summary[l]["compose_given_retrieval"]
            print(f"  {l:<10} recall {summary[l]['recall']:.2f}  fuzzy {summary[l]['fuzzy']:.2f}"
                  f"  compose|retr✓ {cg:.2f}" if cg is not None else f"  {l}")
    for k in klist:
        if f"topic@{k}" in summary and f"dense@{k}" in summary:
            d = summary[f"topic@{k}"]["fuzzy"] - summary[f"dense@{k}"]["fuzzy"]
            print(f"Gate B @k={k} (topic > dense): topic {summary[f'topic@{k}']['fuzzy']:.2f} "
                  f"vs dense {summary[f'dense@{k}']['fuzzy']:.2f}  Δ={d:+.2f}  "
                  f"→ {'topic helps' if d > 0 else 'topic ≈/< dense (management-only)'}")
    iter_labs = [l for l in labels if l.startswith("iter")]
    if iter_labs and dense_labs:
        il, dl = iter_labs[0], f"dense@{klist[0]}"
        if dl in summary:
            print(f"iter vs {dl}: recall {summary[dl]['recall']:.2f}→{summary[il]['recall']:.2f}, "
                  f"fuzzy {summary[dl]['fuzzy']:.2f}→{summary[il]['fuzzy']:.2f}  "
                  f"(iter avg_n {summary[il]['avg_n']:.1f})")

    if args.diagnostic:
        for lab in [l for l in labels if l.startswith("dense")]:
            full = [r for r in records
                    if r["policies"][lab]["recall"] == 1.0 and "single_best" in r["policies"][lab]]
            if not full:
                continue
            sb = sum(r["policies"][lab]["single_best"] for r in full) / len(full)
            cp = sum(r["policies"][lab]["fuzzy"] for r in full) / len(full)
            print(f"DIAGNOSTIC {lab} (full-retrieval, {len(full)}q): best-single {sb:.2f} vs "
                  f"compose {cp:.2f}  Δ={cp - sb:+.2f}  → "
                  f"{'composition chains (fidelity ok)' if cp - sb > 0.15 else 'single≈compose: fidelity/reasoning-bound, not dilution'}")

    if args.show:
        title_of = lambda s: id2entry.get(s, {}).get("title", s)
        show = records if args.show < 0 else records[: args.show]
        print("\n" + "=" * 72 + "\nPER-QUERY OUTPUTS")
        for r in show:
            print(f"\n── {r['id']}  (gold answer: {r['answer']!r})")
            print(f"   Q: {r['question']}")
            if "teacher" in r:
                print(f"   teacher: {r['teacher']['pred']!r}  (f={r['teacher']['fuzzy']:.0f})")
            for lab in labels:
                p = r["policies"][lab]
                titles = ", ".join(title_of(s) for s in p["retrieved"][:4])
                extra = f"  bridge={p['bridge']!r}" if "bridge" in p else ""
                if "answer_q" in p:
                    extra += f"  reframed={p['answer_q']!r}"
                mark = "✓" if p["fuzzy"] else "✗"
                print(f"   [{lab}] r={p['recall']} {mark} pred={p['pred']!r}  «{titles}»{extra}")
                for sg in p.get("singles", []):
                    sm = "✓" if sg["fuzzy"] else "✗"
                    print(f"        single «{title_of(sg['slug'])}» {sm} {sg['pred']!r}")

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "router_results.json"
    out.write_text(json.dumps({"config": vars(args), "summary": summary, "records": records},
                              indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
