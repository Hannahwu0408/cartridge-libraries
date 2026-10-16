"""
eval_fidelity.py — single-cartridge FIDELITY at a given compression (the "does more data make
2x compression work?" probe).

For each article that has a trained cartridge, sample K of its OWN self-study Q/A pairs and ask:
does the cartridge ALONE reproduce the article-grounded answer? We score:
  • cartridge  vs gold  (fuzzy/containment)  — single-cartridge fidelity  ← the number we care about
  • teacher    vs gold  (article in context) — the ceiling for this data
The self-study "gold" answer was itself generated with the article in context, so teacher ≈ 1.0 and
the cartridge's shortfall is exactly the information lost to compression. Run on two cartridge sets
(e.g. old 60-conv/article vs new CAS-style) to read whether more data lifts fidelity at fixed len.

Usage
-----
python scripts/eval_fidelity.py \
    --articles-file data/2wiki_rt/articles.jsonl \
    --conversations-file data/2wiki_rt/selfstudy_big_all.jsonl \
    --cartridge-dir cartridges/2wiki_joint --cartridge-len 128 \
    --model-id Qwen/Qwen3-4B --per-article 8
"""
from __future__ import annotations

import sys
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cartridge_qwen import get_device, load_model, generate_answer, QwenCartridge  # noqa: E402
from evaluate_squad import containment, fuzzy_containment  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="Single-cartridge fidelity vs full-context teacher")
    ap.add_argument("--articles-file", required=True)
    ap.add_argument("--conversations-file", required=True)
    ap.add_argument("--cartridge-dir", required=True)
    ap.add_argument("--cartridge-len", type=int, default=128)
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--per-article", type=int, default=8, help="self-study Qs sampled per article")
    ap.add_argument("--max-articles", type=int, default=None)
    ap.add_argument("--no-teacher", action="store_true", help="skip the teacher ceiling")
    ap.add_argument("--gen-max-tokens", type=int, default=48)
    ap.add_argument("--results-dir", default="results/fidelity")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")
    model, tokenizer = load_model(args.model_id, device)

    ctx = {a["passage_id"]: a["context"] for a in load_jsonl(args.articles_file)}
    convs_by_pid: dict[str, list[dict]] = {}
    for r in load_jsonl(args.conversations_file):
        if r.get("question") and r.get("answer"):
            convs_by_pid.setdefault(r["passage_id"], []).append(r)

    cart_dir = Path(args.cartridge_dir)
    pids = [p for p in ctx if convs_by_pid.get(p)
            and (cart_dir / p / f"cartridge_{args.cartridge_len}.pt").exists()]
    if args.max_articles:
        pids = pids[: args.max_articles]
    print(f"Scoring {len(pids)} cartridges from {args.cartridge_dir}")

    gen = lambda **kw: generate_answer(model, tokenizer, device=device,
                                       max_new_tokens=args.gen_max_tokens, conversational=True, **kw)

    records, cf, tf = [], [], []
    for i, pid in enumerate(pids):
        c = QwenCartridge.load(cart_dir / pid / f"cartridge_{args.cartridge_len}.pt", device="cpu")
        qs = random.sample(convs_by_pid[pid], min(args.per_article, len(convs_by_pid[pid])))
        for conv in qs:
            q, gold = conv["question"], conv["answer"]
            cp = gen(question=q, cartridge=c)
            cfz = fuzzy_containment(cp, gold)
            rec = {"pid": pid, "q": q, "gold": gold, "cart_pred": cp,
                   "cart_fuzzy": cfz, "cart_cont": containment(cp, gold)}
            cf.append(cfz)
            if not args.no_teacher:
                tp = gen(question=q, passage=ctx[pid])
                rec.update({"teacher_pred": tp, "teacher_fuzzy": fuzzy_containment(tp, gold)})
                tf.append(rec["teacher_fuzzy"])
            records.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(pids)}] running cart-fidelity {sum(cf)/len(cf):.3f}")

    m = lambda x: sum(x) / len(x) if x else float("nan")
    print("\n" + "=" * 60)
    print(f"SINGLE-CARTRIDGE FIDELITY  ({args.cartridge_dir}, len={args.cartridge_len}, "
          f"n={len(records)} Qs over {len(pids)} carts)")
    print(f"  cartridge fuzzy (fidelity): {m(cf):.3f}")
    print(f"  cartridge containment:      {m([r['cart_cont'] for r in records]):.3f}")
    if tf:
        print(f"  teacher fuzzy (ceiling):    {m(tf):.3f}")
        print(f"  fidelity / ceiling:         {m(cf)/m(tf):.2f}"
              if m(tf) else "  (teacher 0)")

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fidelity_results.json").write_text(
        json.dumps({"config": vars(args), "cart_fuzzy": m(cf),
                    "teacher_fuzzy": m(tf) if tf else None, "records": records}, indent=2),
        encoding="utf-8")
    print(f"\nWrote {out / 'fidelity_results.json'}")


if __name__ == "__main__":
    main()
