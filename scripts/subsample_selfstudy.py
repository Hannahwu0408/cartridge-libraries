"""subsample_selfstudy.py — carve a per-article PREFIX of a self-study file.

Used to build the convs/passage ladder for the fidelity grid WITHOUT regenerating anything: the
500/pp and 1000/pp arms are prefixes of the 2000/pp file, so a "more data" contrast varies ONLY the
number of conversations — not the generator, the seed, the mode mix, or which articles are covered.
Any other construction (a fresh run at a lower --per-passage, or a random sample) would change the
question distribution too and confound the comparison.

Prefix, not random sample, is deliberate: the generator emits in mode-cycle order, so a prefix is
also what a genuinely shorter run would have produced.

Usage:
  python scripts/subsample_selfstudy.py IN.jsonl OUT.jsonl --cap 500 [--expect-articles 30]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--cap", type=int, required=True, help="max convs kept per passage_id")
    ap.add_argument("--expect-articles", type=int, default=None,
                    help="abort unless exactly this many articles survive (catches a truncated "
                         "source file before it silently becomes a smaller training set)")
    a = ap.parse_args()

    kept = collections.Counter()
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    with open(a.dst, "w", encoding="utf-8") as out:
        for line in open(a.src, encoding="utf-8"):
            if not line.strip():
                continue
            pid = json.loads(line)["passage_id"]
            if kept[pid] < a.cap:
                kept[pid] += 1
                out.write(line)

    if not kept:
        sys.exit(f"ABORT: {a.src} produced no conversations.")
    print(f"  subsample @{a.cap}/pp: {sum(kept.values())} convs over {len(kept)} articles "
          f"(min/max per article {min(kept.values())}/{max(kept.values())}) → {a.dst}")
    if a.expect_articles is not None and len(kept) != a.expect_articles:
        sys.exit(f"ABORT: expected {a.expect_articles} articles, got {len(kept)} — the source file "
                 f"is incomplete; do NOT train on this.")
    short = [p for p, n in kept.items() if n < a.cap]
    if short:
        print(f"  ⚠️  {len(short)} article(s) had fewer than {a.cap} convs available "
              f"(min {min(kept.values())}): {short[:3]}")


if __name__ == "__main__":
    main()
