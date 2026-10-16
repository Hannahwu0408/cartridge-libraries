"""
evaluate_composition.py — the decisive composition test.

For each 2WikiMultihop question whose answer needs two gold articles A, B, score the gold
answer under:
  no_cartridge          — floor
  cart_A / cart_B       — each single cartridge alone
  compose([A,B])        — gated composition (THE TEST), swept over --temps
  compose([A,random])   — control: is it the *relevant* 2nd cartridge, not just "2 cartridges"?
  teacher               — both gold paragraphs in context (ICL ceiling)

Composition WORKS (→ build routing) iff  compose([A,B]) > max(cart_A, cart_B), approaches
teacher, and beats compose([A,random]) — with gate weights showing both A and B contribute.

Reuses generate_answer / compose / ComposedCartridge (cartridge_qwen) and the SQuAD-style
scorers (evaluate_squad). Cartridges come from train_squad_cartridges over articles.jsonl,
all at one --cartridge-len (composition requires equal lengths).

Usage
-----
python scripts/evaluate_composition.py \
    --articles-file data/2wiki/articles.jsonl --multihop-file data/2wiki/multihop.jsonl \
    --cartridge-dir cartridges/2wiki --cartridge-len 128 --model-id Qwen/Qwen3-4B \
    --temps 0.5,1,2,5 --log-gates --results-dir results/2wiki
"""
from __future__ import annotations

import json
import random
import argparse
import sys
from pathlib import Path
from statistics import mean

import torch

sys.path.insert(0, str(Path(__file__).parent))
from cartridge_qwen import (
    get_device, load_model, generate_answer, compose, QwenCartridge,
    _build_conv_prefix, ComposedCartridgeCache, composed_attention, RelevanceGate,
)
from evaluate_squad import containment, exact_match, f1_score, fuzzy_containment


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@torch.no_grad()
def gate_weights(model, tokenizer, question, cA, cB, device):
    """One forward with a debug composed cache → mean per-cartridge gate weight [wA, wB]."""
    prefix = _build_conv_prefix(question, None, tokenizer)
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    cache = ComposedCartridgeCache([cA, cB], device=device, dtype=model.dtype, debug=True)
    pos = torch.arange(cA.cartridge_len, cA.cartridge_len + len(ids),
                       dtype=torch.long, device=device).unsqueeze(0)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with composed_attention(model, cache):
        model(input_ids=inp, past_key_values=cache, position_ids=pos, use_cache=True)
    w = cache.last_cartridge_weights
    return [float(x) for x in w[0]] if w is not None else None


@torch.no_grad()
def group_weights(model, tokenizer, question, cartridges, device, shared_sink=False, gate=None):
    """Per-cartridge gnorm gate weights, measured AT THE ANSWER-GENERATION POSITION (the last
    prefix token — the one that produces the first answer token), forcing mode='gnorm'.

    Returns a dict with two per-cartridge lists:
      • "gen_mean" — mean weight over (layer × head) at the generation position.
      • "gen_topq" — mean of the TOP-QUARTILE of those (layer × head) weights per cartridge.
    The plain mean over all positions/heads washes out the gate (all cartridges look ~equal on the
    many neutral question tokens); measuring at the generation position and taking the top quartile
    surfaces the specific heads doing the hop — where a working gate actually separates the answer
    cartridge from distractors (Phase 11 gate diagnostic)."""
    prefix = _build_conv_prefix(question, None, tokenizer)
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    clen = cartridges[0].cartridge_len
    cache = ComposedCartridgeCache(cartridges, device=device, dtype=model.dtype,
                                   debug=True, mode="gnorm", shared_sink=shared_sink,
                                   model=model, tokenizer=tokenizer, gate=gate)
    pos = torch.arange(clen, clen + len(ids), dtype=torch.long, device=device).unsqueeze(0)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with composed_attention(model, cache):
        model(input_ids=inp, past_key_values=cache, position_ids=pos, use_cache=True)
    gp = cache.last_genpos_weights                       # (n_layers, n_cart, n_heads)
    if gp is None:
        return None
    nc = gp.shape[1]
    flat = gp.permute(1, 0, 2).reshape(nc, -1)           # (n_cart, n_layers*n_heads)
    k = max(1, flat.shape[1] // 4)
    topq = flat.topk(k, dim=1).values.mean(dim=1)
    return {"gen_mean": [float(x) for x in flat.mean(dim=1)],
            "gen_topq": [float(x) for x in topq]}


def score(pred, gold):
    return {"cont": containment(pred, gold), "em": exact_match(pred, gold),
            "f1": f1_score(pred, gold), "fuzzy": fuzzy_containment(pred, gold)}


def main():
    ap = argparse.ArgumentParser(description="Composition test on 2WikiMultihop")
    ap.add_argument("--articles-file", default="data/2wiki/articles.jsonl")
    ap.add_argument("--multihop-file", default="data/2wiki/multihop.jsonl")
    ap.add_argument("--cartridge-dir", default="cartridges/2wiki")
    ap.add_argument("--cartridge-len", type=int, default=128)
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--compose-mode", default="concat", choices=["concat", "gated", "gnorm", "both"],
                    help="concat = KV-concatenation (distinct positions); gated = relevance-gate "
                         "+ 50/50 blend; gnorm = joint-softmax over in-distribution per-cartridge "
                         "attentions + context (order-invariant fix for concat's OOD dilution)")
    ap.add_argument("--temps", default="0.5,1,2,5", help="cartridge_temp sweep (gated mode only)")
    ap.add_argument("--gen-max-tokens", type=int, default=48)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--log-gates", action="store_true", help="Also record per-cartridge gate weights")
    ap.add_argument("--diagnostic", action="store_true",
                    help="Add compose([B,B]) + compose([B,random]) + compose([B,A]) to isolate the "
                         "concat mechanism from content distraction on the strong cartridge")
    ap.add_argument("--shared-sink", action="store_true",
                    help="Add compose_concat_sink = [A,B] concat with a neutral sink prepended "
                         "(tests whether the first cartridge's slot-0 injects its content)")
    ap.add_argument("--ncopies", action="store_true",
                    help="N-copies overlap/sink control: add compose([B]*n) for n=2,3,4 scored on "
                         "B's OWN probe (cart_B answers it alone). Degradation ⇒ gnorm overlap is "
                         "lossy; losslessness ⇒ the k≥3 collapse is the gate mis-weighting distractors.")
    ap.add_argument("--scaling-selection", action="store_true",
                    help="SELECTION scaling (CAS's actual setting): compose [B, N distractors] and "
                         "ask B's OWN probe (single-cartridge, no synthesis). Tests whether joint "
                         "training taught the model to attend to the 1 relevant cart amid clutter, "
                         "separate from the harder 2-hop synthesis scaling.")
    ap.add_argument("--gate", default=None,
                    help="path to a trained RelevanceGate (gnorm only): adds a learned per-cartridge "
                         "relevance weight bias so distractors are downweighted (Phase 11 S1)")
    ap.add_argument("--gnorm-shared-sink", action="store_true",
                    help="Use ONE shared neutral sink for gnorm compositions (strip each "
                         "cartridge's slot-0). Tests whether removing N competing sinks unflattens "
                         "the gate and raises the k≥3 ceiling. Applies to gnorm compose + scaling.")
    ap.add_argument("--scaling-distractors", default=None,
                    help="Distractor-scaling test: comma-sep N values (e.g. '0,1,2,3,4'). For each, "
                         "compose the answer pair [A,B] with N random distractor cartridges prepended "
                         "(OOD front for concat; order-invariant for gnorm) → compose_scale{N}. "
                         "Measures how fast synthesis decays with retrieval clutter. Uses --compose-mode.")
    ap.add_argument("--results-dir", default="results/2wiki")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    temps = [float(t) for t in args.temps.split(",")]
    scaling_Ns = ([int(x) for x in args.scaling_distractors.split(",")]
                  if args.scaling_distractors else None)

    device = get_device()
    print(f"Device: {device}")
    model, tokenizer = load_model(args.model_id, device)
    gate = RelevanceGate.load(args.gate, device=device) if args.gate else None
    if gate is not None:
        print(f"Loaded RelevanceGate from {args.gate}")

    articles = {a["passage_id"]: a for a in load_jsonl(Path(args.articles_file))}
    questions = load_jsonl(Path(args.multihop_file))
    if args.max_questions:
        questions = questions[: args.max_questions]

    # Load every cartridge referenced by the questions.
    cart_dir = Path(args.cartridge_dir)
    carts: dict[str, QwenCartridge] = {}
    def load_cart(slug):
        if slug not in carts:
            p = cart_dir / slug / f"cartridge_{args.cartridge_len}.pt"
            carts[slug] = QwenCartridge.load(p, device=device) if p.exists() else None
        return carts[slug]

    gen = lambda **kw: generate_answer(model, tokenizer, device=device,
                                       max_new_tokens=args.gen_max_tokens, conversational=True, **kw)

    records = []
    for qi, q in enumerate(questions):
        a, b = q["article_slugs"]
        cA, cB = load_cart(a), load_cart(b)
        if cA is None or cB is None:
            print(f"  skip q{qi} ({q['id']}): missing cartridge for {a if cA is None else b}")
            continue
        gold = q["answer"]
        textA, textB = articles[a]["context"], articles[b]["context"]

        rec = {"id": q["id"], "type": q["type"], "question": q["question"], "answer": gold,
               "preds": {}, "scores": {}}

        def run(name, question=None, gold_override=None, **kw):
            pred = gen(question=question if question is not None else q["question"], **kw)
            rec["preds"][name] = pred
            rec["scores"][name] = score(pred, gold_override if gold_override is not None else gold)

        run("no_cartridge")
        run("cart_A", cartridge=cA)
        run("teacher_A", passage=textA)          # single-article teacher → cart_A fidelity ref
        run("cart_B", cartridge=cB)
        run("teacher_B", passage=textB)          # single-article teacher → cart_B fidelity ref
        run("teacher", passage=textA + "\n\n" + textB)
        # single-hop fidelity probes (planted synthesis test): does each cartridge hold
        # its OWN fact? Scored vs the probe answer, not the bridge answer. probe_A tests
        # the hop-1 link in A; probe_B tests the hop-2 target in B. Only pairs where BOTH
        # probes pass but BOTH singles fail on the bridge Q isolate genuine synthesis.
        pa, pb = q.get("probe_a"), q.get("probe_b")
        if pa:
            run("probe_A", question=pa["question"], gold_override=pa["answer"], cartridge=cA)
        if pb:
            run("probe_B", question=pb["question"], gold_override=pb["answer"], cartridge=cB)
        # random control: a cartridge from a different question's article
        others = [s for s in articles if s not in (a, b) and load_cart(s) is not None]
        cR = load_cart(random.choice(others)) if others else None

        if args.compose_mode in ("concat", "both"):
            run("compose_concat", cartridge=compose([cA, cB], mode="concat"))
            if cR is not None:
                run("compose_concat_rand", cartridge=compose([cA, cR], mode="concat"))
            if args.shared_sink:
                # Same [A,B] concat but with a neutral attention sink prepended → tests
                # whether the first cartridge's slot-0 (the sink) was injecting its content.
                run("compose_concat_sink", cartridge=compose([cA, cB], mode="concat"),
                    concat_shared_sink=True)
            if args.diagnostic:
                # Ordering test: answer cartridge (B) in the first/sink slot vs the last/in-dist slot.
                run("compose_concat_BA", cartridge=compose([cB, cA], mode="concat"))
                # Isolate mechanism vs content distraction on the typically-strong cartridge (B):
                #   selfBB ≈ cart_B  → overlapping-concat preserves a duplicated cartridge (mechanism ok)
                #   selfBB <  cart_B → overlapping-concat is inherently lossy (rework needed)
                #   B+rand <  cart_B → adding ANY distractor degrades a strong cartridge
                run("compose_concat_BB", cartridge=compose([cB, cB], mode="concat"))
                if cR is not None:
                    run("compose_concat_B_rand", cartridge=compose([cB, cR], mode="concat"))
        if args.compose_mode in ("gnorm", "both"):
            # joint-softmax over in-distribution per-cartridge attentions + context.
            ss = args.gnorm_shared_sink
            run("compose_gnorm", cartridge=compose([cA, cB], mode="gnorm", shared_sink=ss, gate=gate))
            if cR is not None:
                run("compose_gnorm_rand",
                    cartridge=compose([cA, cR], mode="gnorm", shared_sink=ss, gate=gate))
            if args.diagnostic:
                # order-invariance check: gnorm should give BA ≈ AB (unlike concat).
                run("compose_gnorm_BA", cartridge=compose([cB, cA], mode="gnorm", shared_sink=ss, gate=gate))
                run("compose_gnorm_BB", cartridge=compose([cB, cB], mode="gnorm", shared_sink=ss, gate=gate))
        if args.compose_mode in ("gated", "both"):
            for t in temps:
                run(f"compose_gated@{t}", cartridge=compose([cA, cB], mode="gated"), cartridge_temp=t)
            if cR is not None:
                run("compose_gated_rand", cartridge=compose([cA, cR], mode="gated"),
                    cartridge_temp=temps[len(temps) // 2])
            if args.log_gates:
                rec["gate_weights"] = gate_weights(model, tokenizer, q["question"], cA, cB, device)

        if args.ncopies and pb:
            # Overlap/sink control: N IDENTICAL copies of the ANSWER cartridge B, scored on B's
            # OWN probe (cart_B answers it alone = probe_B). Zero content interference, so any
            # drop = gnorm's overlap/sink mechanics being lossy; losslessness ⇒ the k≥3 collapse
            # is the gate mis-weighting distractors (mechanical, targeted by the shared-sink fix).
            cmode = "gnorm" if args.compose_mode in ("gnorm", "both") else "concat"
            for n in (2, 3, 4):
                run(f"copiesB{n}", question=pb["question"], gold_override=pb["answer"],
                    cartridge=compose([cB] * n, mode=cmode, shared_sink=args.gnorm_shared_sink))

        if scaling_Ns is not None:
            # Distractor scaling: answer pair [A,B] + N random distractors. Distractors go
            # FIRST (OOD front for concat; the answer pair keeps the last/in-dist slots);
            # gnorm is order-invariant so placement is moot. Mechanism = --compose-mode.
            smode = "gnorm" if args.compose_mode == "gnorm" else "concat"
            ss = args.gnorm_shared_sink and smode == "gnorm"
            for N in scaling_Ns:
                if N > len(others):
                    continue
                ds = [load_cart(s) for s in random.sample(others, N)]
                comps = [*ds, cA, cB]
                run(f"compose_scale{N}",
                    cartridge=compose(comps, mode=smode, shared_sink=ss,
                                      gate=gate if smode == "gnorm" else None))
                # Per-group gate weights at this N: does the answer pair keep its mass, or does
                # the gate flatten toward uniform (each distractor stealing ~equal weight)?
                if args.log_gates and smode == "gnorm":
                    gw = group_weights(model, tokenizer, q["question"], comps, device,
                                       shared_sink=ss, gate=gate)
                    if gw is not None:
                        gm, gt = gw["gen_mean"], gw["gen_topq"]
                        avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
                        rec.setdefault("scale_weights", {})[str(N)] = {
                            "ans_mean": avg(gm[N:N + 2]), "dist_mean": avg(gm[:N]),
                            "ans_topq": avg(gt[N:N + 2]), "dist_topq": avg(gt[:N]),
                            "per_cart_mean": gm, "per_cart_topq": gt,
                        }
                # SELECTION scaling (CAS's setting): B alone among N distractors, on B's own probe.
                if args.scaling_selection and pb:
                    run(f"select_scale{N}", question=pb["question"], gold_override=pb["answer"],
                        cartridge=compose([*ds, cB], mode=smode))

        records.append(rec)
        comp_name = {"concat": "compose_concat", "gnorm": "compose_gnorm",
                     "gated": f"compose_gated@{temps[0]}"}.get(args.compose_mode, "compose_concat")
        cv = rec["scores"].get(comp_name, {}).get("cont", float("nan"))
        print(f"q{qi} [{q['type']}] gold={gold!r:28.28} "
              f"A={rec['scores']['cart_A']['cont']:.0f} B={rec['scores']['cart_B']['cont']:.0f} "
              f"compose={cv:.0f} teacher={rec['scores']['teacher']['cont']:.0f}"
              + (f"  gate={rec.get('gate_weights')}" if rec.get('gate_weights') else ""))

    if not records:
        print("No questions evaluated (cartridges missing?).")
        return

    # ── Aggregate (condition set derived from what actually ran) ────────────────────
    seen = []
    for r in records:
        for c in r["scores"]:
            if c not in seen:
                seen.append(c)
    # cart_X next to its single-article teacher_X for an at-a-glance fidelity read
    base = [c for c in ("no_cartridge", "cart_A", "teacher_A", "cart_B", "teacher_B") if c in seen]
    probe = [c for c in ("probe_A", "probe_B") if c in seen]
    comp = [c for c in seen if c.startswith("compose") and "rand" not in c]
    rand = [c for c in seen if c.startswith("compose") and "rand" in c]
    conds = base + probe + comp + rand + (["teacher"] if "teacher" in seen else [])

    def agg(cond, metric):
        vals = [r["scores"][cond][metric] for r in records if cond in r["scores"]]
        return mean(vals) if vals else float("nan")

    print(f"\n{'='*72}\n  COMPOSITION TEST  (n={len(records)}, mode={args.compose_mode})\n{'='*72}")
    print(f"  {'condition':<22}{'cont':>8}{'fuzzy':>8}{'em':>8}{'f1':>8}")
    for c in conds:
        print(f"  {c:<22}{agg(c,'cont'):>8.3f}{agg(c,'fuzzy'):>8.3f}{agg(c,'em'):>8.3f}{agg(c,'f1'):>8.3f}")

    # headline verdict: best compose condition by containment vs best single
    best_comp = max(comp, key=lambda c: agg(c, "cont")) if comp else None
    compose_best = agg(best_comp, "cont") if best_comp else float("nan")
    best_single = max(agg("cart_A", "cont"), agg("cart_B", "cont"))
    print(f"\n  best compose = {best_comp}  →  {compose_best:.3f}   "
          f"max(single)={best_single:.3f}   teacher={agg('teacher','cont'):.3f}")
    works = best_comp is not None and compose_best > best_single + 1e-9
    print(f"  compose > max(single)? {'YES ✓' if works else 'NO ✗'}"
          + (f"   {rand[0]}={agg(rand[0],'cont'):.3f}" if rand else ""))

    # ── Non-destructive check + synthesis rate (planted test). Use the REAL composition
    # conditions ([A,B] with/without sink), NOT the diagnostic self-compose [B,B] which
    # would inflate the metric. First: is concat non-destructive (compose[A,B] ≈ cart_B)?
    # Then: among pairs where both cartridges hold their fact (probes pass) yet NEITHER
    # single answers the bridge, does compose recover it (= synthesis, not union/OR)?
    if probe:
        def ok(r, c):
            return r["scores"].get(c, {}).get("cont", 0.0) >= 1.0
        real = [c for c in ("compose_concat", "compose_concat_sink", "compose_gnorm") if c in seen]
        print(f"\n  fidelity: probe_A={agg('probe_A','cont'):.3f}  probe_B={agg('probe_B','cont'):.3f}"
              f"  (want ≥0.8 before trusting the verdict)")
        cb = agg("cart_B", "cont")
        for c in real:
            print(f"  non-destructive? {c}={agg(c,'cont'):.3f} vs cart_B={cb:.3f}  "
                  f"({'≈ ok' if agg(c,'cont') >= cb - 0.1 else 'DESTRUCTIVE'})")
        elig = [r for r in records if ok(r, "probe_A") and ok(r, "probe_B")
                and not ok(r, "cart_A") and not ok(r, "cart_B")]
        if elig:
            for c in real:
                syn = sum(ok(r, c) for r in elig)
                print(f"  SYNTHESIS RATE ({c}): {syn}/{len(elig)} of (probeA✓ & probeB✓ & cartA✗ & "
                      f"cartB✗) → ✓  =  {syn/len(elig):.2f}  (≳0.5 ⇒ synthesis; ≈0 ⇒ union-only)")
        else:
            print("  SYNTHESIS RATE: no eligible pairs — need probes to pass AND both singles to "
                  "fail the bridge. If probes are low → fidelity issue (more epochs/capacity); "
                  "if a single already answers the bridge → question leaks (tighten the article).")

        # ── Distractor scaling: synthesis rate vs #distractors (does the answer survive
        # retrieval clutter?). fuzzy-hit-based, on the same both-fail eligible pairs.
        scale_conds = sorted((c for c in seen if c.startswith("compose_scale")),
                             key=lambda c: int(c[len("compose_scale"):]))
        if scale_conds and elig:
            def fok(r, c):
                return r["scores"].get(c, {}).get("fuzzy", 0.0) >= 1.0
            print(f"\n  DISTRACTOR SCALING (synth-rate on {len(elig)} both-fail pairs, "
                  f"mechanism={'gnorm' if args.compose_mode=='gnorm' else 'concat'}):")
            for c in scale_conds:
                n = c[len("compose_scale"):]
                syn = sum(fok(r, c) for r in elig)
                print(f"    N={n:>2} distractors → {syn}/{len(elig)} = {syn/len(elig):.2f}"
                      f"   (exact={agg(c,'cont'):.2f}  fuzzy={agg(c,'fuzzy'):.2f})")

    # ── Selection scaling (CAS's setting): does B answer its OWN probe among N distractors? ──
    sel_conds = sorted((c for c in seen if c.startswith("select_scale")),
                       key=lambda c: int(c[len("select_scale"):]))
    if sel_conds:
        pbv = agg("probe_B", "cont")
        print(f"\n  SELECTION SCALING (B alone + N distractors, on B's own probe; probe_B={pbv:.2f}) — "
              f"pure attend-to-the-1-relevant, CAS-style:")
        for c in sel_conds:
            n = c[len("select_scale"):]
            print(f"    N={n:>2} distractors → cont={agg(c,'cont'):.2f}  fuzzy={agg(c,'fuzzy'):.2f}")

    # ── N-copies control: is gnorm lossless on identical content? (mechanical vs fundamental) ──
    ncopy_conds = sorted(c for c in seen if c.startswith("copiesB"))
    if ncopy_conds:
        pbv = agg("probe_B", "cont")
        print(f"\n  N-COPIES CONTROL (compose([B]*n) on B's own probe vs probe_B={pbv:.3f} — "
              f"isolates overlap/sink loss from content interference):")
        for c in ncopy_conds:
            v = agg(c, "cont")
            tag = "≈ lossless (overlap clean → collapse is the gate, fixable)" if v >= pbv - 0.1 \
                else "DEGRADES (overlap/sink lossy even for identical content)"
            print(f"    {c:<20} cont={v:.3f}  {tag}")

    # ── Gate weights AT THE ANSWER-GENERATION POSITION: does the gate separate answer from
    # distractor? The MEAN over heads is misleading (washes out the decisive heads); the
    # TOP-QUARTILE surfaces the heads doing the hop, and its answer-vs-distractor GAP is the
    # honest "is the gate discriminating" signal. ──
    if args.log_gates and scaling_Ns is not None and any("scale_weights" in r for r in records):
        print(f"\n  GATE WEIGHTS AT ANSWER-GENERATION POSITION (mean washes out the gate; "
              f"top-quartile GAP surfaces the decisive heads):")
        for N in scaling_Ns:
            ws = [r["scale_weights"][str(N)] for r in records
                  if r.get("scale_weights", {}).get(str(N))]
            if not ws:
                continue
            am, dm = mean(w["ans_mean"] for w in ws), mean(w["dist_mean"] for w in ws)
            at, dt = mean(w["ans_topq"] for w in ws), mean(w["dist_topq"] for w in ws)
            gap = at - dt
            tag = "" if not N else (f"  gap={gap:+.3f} "
                                    + ("← gate DISCRIMINATES" if gap > 0.05 else "(flat)"))
            print(f"    N={N:>2}: mean ans={am:.3f} dist={dm:.3f}  |  "
                  f"top-quartile ans={at:.3f} dist={dt:.3f}{tag}")
    if args.log_gates:
        gws = [r["gate_weights"] for r in records if r.get("gate_weights")]
        if gws:
            mA, mB = mean(g[0] for g in gws), mean(g[1] for g in gws)
            print(f"  mean gate weights: A={mA:.2f}  B={mB:.2f}  (both >0.2 ⇒ both contribute)")

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "composition_results.json"
    out_path.write_text(json.dumps({"config": vars(args), "records": records}, indent=2,
                                    ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
