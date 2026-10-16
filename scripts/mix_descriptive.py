"""
mix_descriptive.py — build the budget-matched DESCRIPTIVE-addressability self-study for Phase 12-J (2Wiki).

General cap-and-fill mixer (the 2Wiki analog of the synth-specific mix_relational.py, which regex-parses
"apprenticed to the master artisan"). For EVERY article: emit round(frac · C_p) descriptive convs + fill to
C_p with base convs, where C_p = that passage's base-pool conv count. So the output has the SAME per-passage
count as the base pool → n_single identical → n_bridge_per_epoch and steps_per_epoch identical in train_joint
→ the ONLY difference vs the base pool is that a `frac` fraction of each cartridge's self-study is
descriptor-framed instead of name-framed. Isolates FRAMING from training budget.

Descriptive convs are placed FIRST (like mix_relational) so they are never the ones dropped when C_p is small.
If a passage has fewer than round(frac·C_p) descriptive convs available, the shortfall is filled with base
(per-passage total stays exactly C_p either way).

Usage:
  python scripts/mix_descriptive.py \
      --base data/2wiki_v2/selfstudy_v2_base.jsonl \
      --descriptive data/2wiki_v2/selfstudy_v2_desc.jsonl \
      --out data/2wiki_v2/selfstudy_v2_mix.jsonl --frac 0.4
"""
import json
import argparse
import collections
from pathlib import Path


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base self-study jsonl (defines the per-passage budget C_p)")
    ap.add_argument("--descriptive", required=True, help="descriptive (--modes relational) self-study jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.4,
                    help="fraction of each passage's budget to REPLACE with descriptive convs")
    a = ap.parse_args()

    base, desc = load(a.base), load(a.descriptive)
    base_by = collections.defaultdict(list)
    for c in base:
        base_by[c["passage_id"]].append(c)
    desc_by = collections.defaultdict(list)
    for c in desc:
        desc_by[c["passage_id"]].append(c)

    out, n_desc_total, short = [], 0, 0
    for pid, bconvs in base_by.items():
        C = len(bconvs)                              # this passage's base budget (match exactly)
        want = round(a.frac * C)
        d = desc_by.get(pid, [])[:want]              # descriptive first, capped at the target
        if len(d) < want:
            short += 1
        take_base = C - len(d)                       # fill to C with base → per-passage total == C
        out += d + bconvs[:take_base]
        n_desc_total += len(d)

    Path(a.out).write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in out), encoding="utf-8")
    print(f"wrote {len(out)} convs ({n_desc_total} descriptive, frac={a.frac}, "
          f"{short} passages short on descriptive) → {a.out}")
    print(f"per-passage totals match base: {all(len([c for c in out if c['passage_id']==p]) == len(base_by[p]) for p in base_by)}")


if __name__ == "__main__":
    main()
