"""summarize_fidelity_grid.py — read the pp x epochs fidelity grid, and first decide whether the
metric may be trusted to pick the recipe at all.

TWO OUTPUTS, IN THIS ORDER.

1. PROXY VALIDATION. The grid is only usable if single-cartridge fidelity ranks arms the way the
   blind-judged chain accuracy did (orig 0.206 < C 0.235 < C8 0.265 < A 0.353 on 34 questions).
   The failure mode to catch is specific and already observed once: C8 drove training loss 3.4x
   below C's and bought +1 question. If fidelity likewise crowns C8, it is measuring FITTING rather
   than retained, usable knowledge — and the grid must not choose the recipe. Reported as Spearman
   rank correlation over the four arms that have both numbers, plus the explicit C8-vs-A ordering,
   which is the diagnostic pair (compute-matched: 239,760 vs 239,676 gradient steps).

2. THE GRID. fidelity / teacher-ceiling per (convs-per-passage x epochs) cell. The ceiling is
   computed once (the `orig` run) because the teacher answer is cartridge-independent and greedy,
   so it is identical across arms; every other arm runs --no-teacher and takes the ratio from here.

The 4000/pp cells cover only the first 10 articles, so they are compared on a 10-article restriction
of every arm — never against the 30-article numbers.

Usage:
  python scripts/summarize_fidelity_grid.py --grid-dir results/fidelity_grid
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# arm dir -> (convs per passage, epochs); 'orig' is the reference pool, outside the grid
CELLS = {
    "500pp_4ep": (500, 4),   "500pp_8ep": (500, 8),
    "1000pp_4ep": (1000, 4), "1000pp_8ep": (1000, 8),
    "2000pp_4ep": (2000, 4), "2000pp_8ep": (2000, 8),
    "4000pp_4ep": (4000, 4), "4000pp_8ep": (4000, 8),
}
# audited chain accuracy from the blind judge (results/claude_oracle/judge_all.json), n=34
AUDITED = {"orig": 0.206, "1000pp_4ep": 0.235, "1000pp_8ep": 0.265, "2000pp_4ep": 0.353}


def load(grid: Path, arm: str):
    p = grid / arm / "fidelity_results.json"
    return json.loads(p.read_text()) if p.exists() else None


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n * n - 1)) if n > 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", default="results/fidelity_grid")
    ap.add_argument("--top-articles", type=int, default=10,
                    help="how many articles the 4000/pp cells cover")
    a = ap.parse_args()
    grid = Path(a.grid_dir)

    runs = {arm: d for arm in ["orig"] + list(CELLS) if (d := load(grid, arm))}
    if not runs:
        raise SystemExit(f"No fidelity_results.json under {grid} — has the driver finished A3?")

    # the ceiling lives in the one run that was NOT given --no-teacher
    ceiling = next((r["teacher_fuzzy"] for r in runs.values() if r.get("teacher_fuzzy")), None)
    if ceiling is None:
        raise SystemExit("No teacher ceiling in any run — re-score one arm without --no-teacher.")

    def ratio(arm, slugs=None):
        recs = runs[arm]["records"]
        if slugs is not None:
            recs = [r for r in recs if r["pid"] in slugs]
        if not recs:
            return None
        return sum(r["cart_fuzzy"] for r in recs) / len(recs) / ceiling

    # ── 1. proxy validation ───────────────────────────────────────────────────────────────
    print("=" * 72)
    print("1. PROXY VALIDATION — does fidelity rank the arms the way audited accuracy did?")
    both = [k for k in AUDITED if k in runs]
    print(f"  {'arm':<12}{'audited acc':>13}{'fidelity/ceiling':>20}")
    for k in sorted(both, key=lambda x: AUDITED[x]):
        print(f"  {k:<12}{AUDITED[k]:>13.3f}{ratio(k):>20.3f}")
    if len(both) >= 3:
        rho = spearman([AUDITED[k] for k in both], [ratio(k) for k in both])
        print(f"\n  Spearman rank correlation: {rho:+.2f}  (n={len(both)} arms)")
        c8, a4 = ratio("1000pp_8ep"), ratio("2000pp_4ep")
        if c8 is not None and a4 is not None:
            diag = ("A > C8 — consistent with the accuracy result (more unique data beats more "
                    "passes at matched compute)" if a4 > c8 else
                    "C8 >= A — WARNING: fidelity crowns the arm that was NULL on accuracy. This is "
                    "the fitting-vs-knowledge failure; do NOT let the grid pick the recipe.")
            print(f"  diagnostic pair (compute-matched): C8 {c8:.3f} vs A {a4:.3f} → {diag}")
        print(f"  VERDICT: {'usable proxy' if rho >= 0.8 else 'WEAK — treat the grid as indicative only'}")

    # ── 2. the grid ───────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("2. GRID — fidelity / teacher-ceiling   (30 articles unless marked)")
    pps = sorted({pp for pp, _ in CELLS.values()})
    print(f"  {'convs/passage':<15}{'4 epochs':>12}{'8 epochs':>12}   n_articles")
    for pp in pps:
        cells = [f"{pp}pp_{e}ep" for e in (4, 8)]
        vals = [ratio(c) if c in runs else None for c in cells]
        n_art = len({r["pid"] for r in runs[cells[0]]["records"]}) if cells[0] in runs else 0
        cols = "".join(f"{v:>12.3f}" if v is not None else f"{'—':>12}" for v in vals)
        print(f"  {pp:<15}{cols}   {n_art or '—'}")
    if "orig" in runs:
        print(f"  {'(orig ref)':<15}{ratio('orig'):>12.3f}{'':>12}   "
              f"{len({r['pid'] for r in runs['orig']['records']})}"
              f"   ← 4B-generated data, probed with 8B golds: style confound, not a clean cell")

    # ── 3. 4000/pp comparison, restricted to the same articles ────────────────────────────
    top = [c for c in ("4000pp_4ep", "4000pp_8ep") if c in runs]
    if top:
        slugs = {r["pid"] for r in runs[top[0]]["records"]}
        print("\n" + "=" * 72)
        print(f"3. TOP-END: every arm restricted to the {len(slugs)} articles the 4000/pp cells cover")
        print(f"  {'cell':<15}{'fidelity/ceiling':>20}")
        for arm in [c for c in CELLS if c in runs]:
            r = ratio(arm, slugs)
            if r is not None:
                print(f"  {arm:<15}{r:>20.3f}")
        print("\n  READ: still climbing at 4000 ⇒ the campaign should buy convs/passage before "
              "articles.\n        Flat by 2000 ⇒ take the cheapest pp within noise (a 2-4x cost "
              "difference in Phase B).")


if __name__ == "__main__":
    main()
