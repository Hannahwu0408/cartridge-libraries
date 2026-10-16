"""
resolution_gap_test.py — isolate the intermediate-RESOLUTION gap in 2-hop composition.

Finding under test: 2Wiki compose fails not because it can't retrieve from both cartridges, but
because in the one-shot forward it never instantiates the intermediate entity ("X's father = William
Douglas") — so it never forms the internal query "William Douglas birthplace". Chain works because it
resolves the intermediate explicitly, then re-queries.

Per bridge question, on the SAME gold pair (gA, gB), three conditions:
  (a) compose([gA,gB], gnorm) + the ORIGINAL bridge query           — the failure
  (b) SINGLE answer-cartridge + ORACLE-resolved query              — resolution + no composition
  (c) compose([gA,gB], gnorm) + ORACLE-resolved query              — resolution + composition
"oracle-resolved" = reframe the question about a gold's ARTICLE TITLE (hand it the intermediate);
we try both gold titles and take the max (best-case oracle).
  • (c) ≈ (b) ≫ (a)  → composition retrieves fine; the gap is resolving the intermediate reference.
  • (c) also low      → deeper retrieval/composition problem.
Also flags whether the answer article cross-references the subject (whether the descriptor the query
uses is present in the answer cartridge's OWN article → scope for the Step-2 addressability fix).

Usage
-----
python scripts/resolution_gap_test.py \
    --multihop-file data/2wiki_rt/multihop.jsonl --articles-file data/2wiki_rt/articles.jsonl \
    --cartridge-dir cartridges/2wiki_joint --cartridge-len 128 --model-id Qwen/Qwen3-4B
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cartridge_qwen import get_device, load_model, generate_answer, compose, QwenCartridge  # noqa: E402
from evaluate_squad import fuzzy_containment  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="Isolate the intermediate-resolution gap in composition")
    ap.add_argument("--multihop-file", required=True)
    ap.add_argument("--articles-file", required=True)
    ap.add_argument("--cartridge-dir", required=True)
    ap.add_argument("--cartridge-len", type=int, default=128)
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--compose-mode", default="gnorm", choices=["gnorm", "concat"])
    ap.add_argument("--gen-max-tokens", type=int, default=48)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--results-dir", default="results/resolution_gap")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    model, tokenizer = load_model(args.model_id, device)
    arts = {a["passage_id"]: a for a in load_jsonl(args.articles_file)}
    title = {s: a.get("title", s) for s, a in arts.items()}
    cart_dir = Path(args.cartridge_dir)
    cache: dict[str, QwenCartridge] = {}

    def load_cart(slug):
        if slug not in cache:
            p = cart_dir / slug / f"cartridge_{args.cartridge_len}.pt"
            cache[slug] = QwenCartridge.load(p, device="cpu") if p.exists() else None
        return cache[slug]

    gen = lambda **kw: generate_answer(model, tokenizer, device=device,
                                       max_new_tokens=args.gen_max_tokens, conversational=True, **kw)

    def reframe(q_text, entity):
        """Rewrite the bridge question as a direct question about the (oracle) intermediate entity."""
        instr = (f'Rewrite the question below so it asks DIRECTLY about "{entity}", keeping the EXACT '
                 f'same thing asked (same who/what/where/when/why and attribute); only replace the '
                 f'described entity with "{entity}".\n\nOriginal: "{q_text}"\n\nReply with ONLY the '
                 f'rewritten question.')
        r = gen(question=instr)
        return (r.strip().splitlines()[0].strip().strip('"') if r.strip() else q_text) or q_text

    qs = load_jsonl(args.multihop_file)
    if args.max_questions:
        qs = qs[: args.max_questions]

    records, A, B, C, D = [], [], [], [], []
    for i, q in enumerate(qs):
        slugs = [s for s in q.get("article_slugs", []) if load_cart(s) is not None]
        if len(slugs) != 2:
            continue
        gold = q["answer"]
        carts = [load_cart(s) for s in slugs]
        composed = compose(carts, mode=args.compose_mode)

        # (a) compose + original bridge query
        a_pred = gen(question=q["question"], cartridge=composed)
        fa = fuzzy_containment(a_pred, gold)

        # (d) SINGLE cart + original bridge query (no oracle) — the ADDRESSABILITY metric: does the
        #     answer cartridge answer the bridge question DIRECTLY (descriptor-addressable)? best over golds.
        fd = max(fuzzy_containment(gen(question=q["question"], cartridge=c), gold) for c in carts)

        # (b)/(c): oracle-resolve about each gold title, take the best
        best_b, best_c, best_rq, best_g = 0.0, 0.0, "", slugs[0]
        for s, c in zip(slugs, carts):
            rq = reframe(q["question"], title[s])
            fb = fuzzy_containment(gen(question=rq, cartridge=c), gold)              # single
            fc = fuzzy_containment(gen(question=rq, cartridge=composed), gold)       # compose
            if max(fb, fc) > max(best_b, best_c):
                best_b, best_c, best_rq, best_g = fb, fc, rq, s
            else:
                best_b, best_c = max(best_b, fb), max(best_c, fc)

        # descriptor-in-own-article proxy: does each gold's article mention the OTHER's title?
        xref = {s: (title[slugs[1 - j]].split(",")[0].lower() in arts[s]["context"].lower())
                for j, s in enumerate(slugs)}

        A.append(fa); B.append(best_b); C.append(best_c); D.append(fd)
        records.append({"id": q.get("id", i), "question": q["question"], "gold": gold,
                        "a_compose_bridge": fa, "d_single_bridge": fd,
                        "b_single_oracle": best_b, "c_compose_oracle": best_c,
                        "oracle_reframe": best_rq, "a_pred": a_pred, "xref": xref})
        print(f"[{i+1}/{len(qs)}] a={fa:.0f} d={fd:.0f} b={best_b:.0f} c={best_c:.0f}  {q['question'][:56]}")

    m = lambda x: sum(x) / len(x) if x else float("nan")
    print(f"\n{'='*70}\nRESOLUTION-GAP  ({args.cartridge_dir}, n={len(A)})")
    print(f"  (a) compose + bridge query     : {m(A):.3f}")
    print(f"  (d) single  + bridge query     : {m(D):.3f}   ← ADDRESSABILITY (no oracle): can a single "
          f"cartridge answer the bridge query directly?")
    print(f"  (b) single  + oracle query     : {m(B):.3f}")
    print(f"  (c) compose + oracle query     : {m(C):.3f}")
    print(f"  → (c)≈(b)≫(a): gap is INTERMEDIATE RESOLUTION.  (d)→(b): descriptive addressability closes it.")
    xr = sum(1 for r in records if any(r["xref"].values()))
    print(f"  answer-article cross-references the subject (fix-scope proxy): {xr}/{len(records)}")

    out = Path(args.results_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "resolution_gap.json").write_text(json.dumps(
        {"config": vars(args), "a": m(A), "d": m(D), "b": m(B), "c": m(C), "records": records}, indent=2),
        encoding="utf-8")
    print(f"\nWrote {out / 'resolution_gap.json'}")


if __name__ == "__main__":
    main()
