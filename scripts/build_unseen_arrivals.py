"""build_unseen_arrivals.py — select GENUINELY-UNSEEN 2Wiki questions for the continual-add Phase 3.

Given a LARGE prepare_2wiki draw (articles_big + multihop_big) and the EXISTING pool's articles, pick the first
N questions whose BOTH gold articles are NOT in the existing pool (truly unseen — no held-out contamination),
and write their articles + questions to an out-dir. These become the "new document arrivals" whose isolated
cartridges we train and add to the library.

Usage:
  python scripts/build_unseen_arrivals.py \
    --big-articles data/2wiki_big/articles.jsonl --big-multihop data/2wiki_big/multihop.jsonl \
    --existing-articles data/2wiki_v2/articles.jsonl --n-questions 12 --out-dir data/2wiki_unseen
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def golds(q):
    return list(q.get("article_slugs") or q.get("gold") or [])[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big-articles", required=True)
    ap.add_argument("--big-multihop", required=True)
    ap.add_argument("--existing-articles", required=True)
    ap.add_argument("--n-questions", type=int, default=12)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    existing = {j["passage_id"] for j in load_jsonl(a.existing_articles)}
    big_art = {j["passage_id"]: j for j in load_jsonl(a.big_articles)}
    big_mh = load_jsonl(a.big_multihop)

    kept_q, kept_slugs = [], []
    seen = set()
    for q in big_mh:
        g = golds(q)
        if len(g) < 2:
            continue
        if any(s in existing for s in g):            # overlaps the existing pool → not unseen
            continue
        if not all(s in big_art for s in g):         # article text must be available to train a cartridge
            continue
        kept_q.append(q)
        for s in g:
            if s not in seen:
                seen.add(s)
                kept_slugs.append(s)
        if len(kept_q) >= a.n_questions:
            break

    if len(kept_q) < a.n_questions:
        print(f"WARNING: only {len(kept_q)} unseen questions available (asked {a.n_questions}) — "
              f"increase the big draw's --max-questions.")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "articles.jsonl").write_text(
        "\n".join(json.dumps(big_art[s], ensure_ascii=False) for s in kept_slugs) + "\n", encoding="utf-8")
    (out / "multihop.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in kept_q) + "\n", encoding="utf-8")
    print(f"wrote {out}/: {len(kept_q)} unseen questions over {len(kept_slugs)} new articles "
          f"(all disjoint from the {len(existing)} existing).")


if __name__ == "__main__":
    main()
