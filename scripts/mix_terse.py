"""mix_terse.py — build the budget-matched MIXED (half-terse) self-study arm for the style-lock
vs compression decomposition.

Why not mix_descriptive.py: that mixer takes desc[:want] + base[:C-want], which is correct when the
two files hold DIFFERENT questions (its original descriptive-framing use). Here the terse file is a
`--reanswer` of the base file — same questions, same order, only the answers differ — so that logic
would emit the same question twice (once terse, once verbose) and DROP the back half of each
passage's questions entirely. That halves question diversity, which collides with the finding that
data quantity is the only lever that moves accuracy: the arm would then differ from the base pool in
two ways at once instead of one.

This mixer instead splits each passage's questions into two DISJOINT halves and takes the terse
answer for one half and the verbose answer for the other. Every question survives exactly once, the
per-passage count is unchanged, so the ONLY variable vs the base pool is the answer style of half
the conversations.

The split is a seeded shuffle, not positional: the generator cycles seed modes, so index order
correlates with seed_type and taking a positional half would systematically bias which modes get
the terse treatment.

SHORT is not hard-coded — it is derived from the data (terse question minus base question) and
asserted identical across all pairs, so this script cannot drift from the generator's constant.

Usage:
  python scripts/mix_terse.py --base data/2wiki_scale/ss_orig150.jsonl \
      --terse data/2wiki_scale/ss_terse150.jsonl \
      --out  data/2wiki_scale/ss_mixed150.jsonl --frac 0.5
"""
from __future__ import annotations

import json
import random
import argparse
import collections
from pathlib import Path


def load_by_passage(path: str) -> dict[str, list[dict]]:
    d: dict[str, list[dict]] = collections.defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            d[r["passage_id"]].append(r)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="original (verbose) self-study jsonl")
    ap.add_argument("--terse", required=True, help="--reanswer --terse-frac 1.0 output of --base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.5, help="fraction of each passage taken terse")
    ap.add_argument("--seed", type=int, default=43)
    a = ap.parse_args()

    B, T = load_by_passage(a.base), load_by_passage(a.terse)
    if set(B) != set(T):
        raise SystemExit(f"ABORT: passage sets differ — base {len(B)}, terse {len(T)}, "
                         f"symmetric difference {len(set(B) ^ set(T))}")

    # Derive SHORT from the data rather than hard-coding it, then require it to be constant.
    suffixes = set()
    for pid in B:
        if len(T[pid]) != len(B[pid]):
            raise SystemExit(f"ABORT: {pid} has {len(T[pid])} terse vs {len(B[pid])} base "
                             f"conversations — the files are not parallel, so a positional "
                             f"pairing would mismatch questions.")
        for x, y in zip(T[pid], B[pid]):
            if not x["question"].startswith(y["question"]):
                raise SystemExit(f"ABORT: question order diverged in {pid}:\n"
                                 f"  base : {y['question'][:80]!r}\n  terse: {x['question'][:80]!r}")
            suffixes.add(x["question"][len(y["question"]):])
    if len(suffixes) != 1:
        raise SystemExit(f"ABORT: terse questions carry {len(suffixes)} distinct suffixes; "
                         f"expected exactly one (the SHORT instruction).")
    short = suffixes.pop()
    print(f"derived SHORT suffix ({len(short)} chars): {short.strip()[:60]!r}...")

    out, n_terse = [], 0
    for pid in sorted(B):
        b, t = B[pid], T[pid]
        idx = list(range(len(b)))
        random.Random(f"{a.seed}:{pid}").shuffle(idx)      # per-passage, order-independent
        terse_at = set(idx[: round(a.frac * len(b))])
        for i in range(len(b)):
            out.append(t[i] if i in terse_at else b[i])
            n_terse += i in terse_at

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    per = collections.Counter(r["passage_id"] for r in out)
    uniq = {(r["passage_id"], r["question"].replace(short, "")) for r in out}
    base_n = sum(len(v) for v in B.values())
    print(f"wrote {len(out)} convs → {a.out}")
    print(f"  terse fraction     : {n_terse}/{len(out)} = {n_terse/len(out):.1%} (target {a.frac:.0%})")
    print(f"  per-passage counts : {min(per.values())}–{max(per.values())} "
          f"(base {min(len(v) for v in B.values())}–{max(len(v) for v in B.values())})")
    print(f"  unique questions   : {len(uniq)} (want {base_n} — proves none duplicated or dropped)")
    ok = (len(out) == base_n and len(uniq) == base_n
          and all(per[p] == len(B[p]) for p in B))
    print(f"  BUDGET MATCHED     : {'YES' if ok else 'NO — do not train on this'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
