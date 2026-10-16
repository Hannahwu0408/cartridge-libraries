"""summarize_qbatch.py — read a finished question-batching sweep and print the decision table.

Split out of probe_question_batching.sh so the analysis can be re-run on saved data without repeating
an hour of GPU work (the inline heredoc version died on a syntax error after a completed sweep, and
the numbers were only recoverable by hand from the log).

Reads results/qbatch_probe/rows.tsv (written per configuration: TAG, convs, gen_seconds, peak_MiB,
saturation_json) plus each saturation sidecar for the duplicate-waste column.

Usage:
  python scripts/summarize_qbatch.py [--dir results/qbatch_probe] [--gate 1.4]
"""
from __future__ import annotations

import argparse
import json
import os

# measured elsewhere in this project, used to project the campaign
DEEP_BASE = 2326 * 1.22   # pool196 rate at answer-batch 8, x the measured batch-32 gain
POD_RATE = 3098 * 1.22
CARTS, PP, EPOCHS, POD_HR = 980, 2000, 4, 0.99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/qbatch_probe")
    ap.add_argument("--gate", type=float, default=1.4)
    a = ap.parse_args()

    rows = []
    for line in open(os.path.join(a.dir, "rows.tsv"), encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[1] == "OOM":
            rows.append((p[0], None, None, None, None))
        elif len(p) == 5 and p[1]:
            rows.append((p[0], int(p[1]), int(p[2]), p[3], p[4]))
    ok = [r for r in rows if r[1]]
    if not ok:
        raise SystemExit(f"no successful configurations in {a.dir}/rows.tsv")
    short = [r for r in ok if r[2] < 60]
    if short:
        print(f"  *** WARNING: {len(short)} configuration(s) ran <60 s of generation — rates "
              f"unreliable, re-run with a larger --per-passage ***\n")

    # baseline is TODAY's configuration, not merely the first row
    todays = [r for r in ok if r[0] == "ab32_r1"] or [ok[0]]
    base = todays[0][1] / todays[0][2] * 3600

    def waste(sat):
        if not sat or not os.path.exists(sat):
            return None
        s = json.load(open(sat))
        par = sum(x.get("parsed", 0) for x in s)
        kep = sum(x.get("kept", 0) for x in s)
        return (1 - kep / par) * 100 if par else None

    base_waste = waste(todays[0][4])
    print(f"{'ansB:rounds':>13}{'convs/h':>10}{'speedup':>9}{'dup-waste':>11}{'vs base':>9}"
          f"{'peak MiB':>10}")
    best = None
    for tag, n, t, mem, sat in rows:
        lbl = tag.replace("ab", "").replace("_r", ":")
        if n is None:
            print(f"{lbl:>13}{'OOM':>10}{'-':>9}{'-':>11}{'-':>9}{'-':>10}")
            continue
        rate = n / t * 3600
        w = waste(sat)
        ws = f"{w:.0f}%" if w is not None else "-"
        dw = f"{w - base_waste:+.0f}pt" if (w is not None and base_waste is not None) else "-"
        print(f"{lbl:>13}{rate:>10,.0f}{rate/base:>8.2f}x{ws:>11}{dw:>9}{mem:>10}")
        if best is None or rate > best[1]:
            best = (lbl, rate, tag, w)
    mult = best[1] / base

    print(f"\nBEST: {best[0]} at {best[1]:,.0f} convs/h = {mult:.3f}x over today (32:1)")
    print("  dup-waste: only the 'vs base' DELTA is meaningful. At any depth most rejected questions")
    print("  are ordinary repeats of the passage's own history; the increase over the 32:1 row is the")
    print("  within-batch penalty (batched rounds cannot see each other's questions). A row that is")
    print("  faster but discards materially more of what it generates is not actually cheaper.")

    convs, steps = CARTS * PP, CARTS * PP * EPOCHS
    print(f"\n500-Q campaign ({CARTS} cartridges x {PP}/pp) at {mult:.2f}x:")
    for pods, lbl in [(0, "1 local GPU"), (1, "local + 1 pod"), (2, "local + 2 pods")]:
        g = (DEEP_BASE + POD_RATE * pods) * mult
        t = (9.3 + 12 * pods) * 3600
        gh, th = convs / g, steps / t
        print(f"  {lbl:<15} gen {gh/24:5.1f} d + train {th/24:4.1f} d = {(gh+th)/24:5.1f} d"
              f"   rent ${(gh+th)*pods*POD_HR:>6,.0f}")
    print(f"\n  NOTE: training does not benefit from any generation speedup — it is ~10 days on one")
    print(f"  GPU regardless, which is what a second GPU actually buys.")

    ab, r = best[0].split(":")
    if mult >= a.gate:
        print(f"\n  GATE (>={a.gate}x): PASS — adopt --answer-batch-size {ab} "
              f"--question-batch-rounds {r}")
        if mult < a.gate * 1.1:
            print(f"  (marginal: {mult:.3f}x against a {a.gate}x gate, one 800-conv run per "
                  f"configuration and no repeats — treat neighbouring rows as ties)")
    else:
        print(f"\n  GATE (>={a.gate}x): FAIL — stop optimising, rent the second GPU")


if __name__ == "__main__":
    main()
