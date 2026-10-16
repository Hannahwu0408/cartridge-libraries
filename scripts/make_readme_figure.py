"""make_readme_figure.py — the README's results figure, generated from the audit files (never hand-drawn).

Two stacked panels on separate axes (never a dual axis):
  A. audited accuracy under a matched controller, both decodings, with the frontier ceiling as a rule
  B. input tokens per question for the two retrieval systems, both decodings

Color follows the entity across both panels: cartridge chain = slot 1, text + same controller = slot 2,
gold-handed teacher = slot 3. Writes a light and a stepped dark SVG for GitHub's <picture> element.

Usage (from the research project directory that holds results/):
  python scripts/make_readme_figure.py \
      --judge results/claude_oracle_scale500/judge_all.json \
      --results results/scale500 --out-dir /path/to/assets
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", base="#c3c2b7", s1="#2a78d6", s2="#eb6834", s3="#1baf7a"),
    "dark":  dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", base="#383835", s1="#3987e5", s2="#d95926", s3="#199e70"),
}
FONT = 'system-ui, -apple-system, &quot;Segoe UI&quot;, Helvetica, Arial, sans-serif'
W, PAD, LABEL_COL, RIGHT = 860, 28, 132, 92          # canvas, padding, row-label column, room for tip labels
BAR, GAP, GROUP_GAP, R = 18, 2, 18, 4                 # bar thickness, surface gap, group gap, data-end radius


def audited(judge_path: Path) -> tuple[dict, int]:
    J = json.loads(judge_path.read_text())
    lab, broken = {}, set()
    for r in J["records"]:
        lab.setdefault(r["system"], {})[r["id"]] = r["verdict"] == "equivalent"
        if r["verdict"] == "gold_problem":
            broken.add(r["id"])
    ids = sorted(set(lab["chain"]) - broken)
    return {s: sum(v[q] for q in ids) / len(ids) for s, v in lab.items()}, len(ids)


def bar(x0, x1, y, h, fill):
    """Square at the baseline, 4px rounded data-end."""
    r = min(R, (x1 - x0) / 2, h / 2)
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} A{r},{r} 0 0 1 {x1:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} A{r},{r} 0 0 1 {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z" fill="{fill}"/>')


def text(x, y, s, fill, size=12, weight=400, anchor="start", tabular=False):
    num = ' font-variant-numeric="tabular-nums"' if tabular else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" font-family="{FONT}"{num}>{s}</text>')


def panel(out, t, y, title, groups, vmax, ticks, fmt, rule=None):
    """groups: [(row_label, [(series_key, value, tip_label), ...]), ...]"""
    x0, x1 = PAD + LABEL_COL, W - PAD - RIGHT
    sx = lambda v: x0 + (x1 - x0) * v / vmax
    out.append(text(PAD, y, title, t["ink"], 13, 600))
    top = y + 22
    h_total = sum(len(g[1]) * (BAR + GAP) - GAP for g in groups) + GROUP_GAP * (len(groups) - 1)
    bottom = top + h_total
    for v in ticks:                                                     # recessive hairline grid
        out.append(f'<line x1="{sx(v):.1f}" y1="{top - 4}" x2="{sx(v):.1f}" y2="{bottom + 4}" '
                   f'stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(text(sx(v), bottom + 20, fmt(v), t["muted"], 11, anchor="middle", tabular=True))
    if rule:                                                            # single reference rule, ink not series
        v, lbl = rule
        out.append(f'<line x1="{sx(v):.1f}" y1="{top - 8}" x2="{sx(v):.1f}" y2="{bottom + 4}" '
                   f'stroke="{t["ink2"]}" stroke-width="1"/>')
        out.append(text(sx(v) - 6, top - 10, lbl, t["ink2"], 11, anchor="end"))
    cy = top
    for row_label, items in groups:
        gh = len(items) * (BAR + GAP) - GAP
        out.append(text(PAD, cy + gh / 2 + 4, row_label, t["ink2"], 12))
        for key, val, tip in items:
            out.append(bar(x0, sx(val), cy, BAR, t[key]))
            out.append(text(sx(val) + 6, cy + BAR / 2 + 4, tip, t["ink"], 12, tabular=True))
            cy += BAR + GAP
        cy += GROUP_GAP - GAP
    out.append(f'<line x1="{x0}" y1="{top - 4}" x2="{x0}" y2="{bottom + 4}" '
               f'stroke="{t["base"]}" stroke-width="1"/>')                 # baseline
    return bottom + 34


def render(t, A, n, tok):
    out = []
    y = PAD + 6
    out.append(text(PAD, y, "Cartridge memory vs. reading the retrieved text", t["ink"], 16, 600))
    out.append(text(PAD, y + 20, f"2WikiMultihopQA · 500 questions over a 2,867-passage corpus · "
                                 f"blind LLM-judge audit, n = {n}", t["ink2"], 12))
    lx, ly = PAD, y + 46                                                # legend: always present, ink text
    for key, lbl in [("s1", "Cartridge chain (ours)"), ("s2", "Retrieved text, same controller"),
                     ("s3", "4B reader handed both gold articles")]:
        out.append(f'<rect x="{lx}" y="{ly - 9}" width="10" height="10" rx="2" fill="{t[key]}"/>')
        out.append(text(lx + 16, ly, lbl, t["ink2"], 12))
        lx += 16 + len(lbl) * 6.0 + 28
    y = ly + 34
    pct = lambda v: f"{v:.2f}".rstrip("0").rstrip(".") if v not in (0, 1) else f"{v:.0f}"
    y = panel(out, t, y, "Accuracy (share judged correct)", [
        ("Verbose answers", [("s1", A["s500_fix_chain_verbose"], f'{A["s500_fix_chain_verbose"]:.3f}'),
                             ("s2", A["s500_lg_text"], f'{A["s500_lg_text"]:.3f}'),
                             ("s3", A["teacher"], f'{A["teacher"]:.3f}')]),
        ("Terse answers", [("s1", A["s500_fix_chain_terse"], f'{A["s500_fix_chain_terse"]:.3f}'),
                           ("s2", A["s500_lg_text_terse"], f'{A["s500_lg_text_terse"]:.3f}'),
                           ("s3", A["s500_teacher_terse"], f'{A["s500_teacher_terse"]:.3f}')]),
    ], 1.0, [0, 0.25, 0.5, 0.75, 1.0], pct, rule=(A["ceiling"], f'Frontier model + gold articles {A["ceiling"]:.3f}'))
    cv, tv, ct, tt = tok["lg_cart"], tok["lg_text"], tok["fix_lg_cart_terse"], tok["lg_text_terse"]
    y = panel(out, t, y + 18, "Input tokens per question", [
        ("Verbose answers", [("s1", cv, f"{cv:.0f}  ·  {tv / cv:.1f}× fewer"), ("s2", tv, f"{tv:.0f}")]),
        ("Terse answers", [("s1", ct, f"{ct:.0f}  ·  {tt / ct:.1f}× fewer"), ("s2", tt, f"{tt:.0f}")]),
    ], 1000, [0, 250, 500, 750, 1000], lambda v: f"{v:,.0f}")
    H = y + 4
    desc = (f"Audited accuracy: cartridge chain {A['s500_fix_chain_verbose']:.3f} verbose and "
            f"{A['s500_fix_chain_terse']:.3f} terse; retrieved text with the same controller "
            f"{A['s500_lg_text']:.3f} and {A['s500_lg_text_terse']:.3f}; reader given both gold articles "
            f"{A['teacher']:.3f} and {A['s500_teacher_terse']:.3f}; frontier ceiling {A['ceiling']:.3f}. "
            f"Input tokens per question: cartridge {cv:.0f} vs text {tv:.0f} verbose, {ct:.0f} vs {tt:.0f} terse.")
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H:.0f}" viewBox="0 0 {W} {H:.0f}" '
            f'role="img" aria-labelledby="t d"><title id="t">Cartridge memory vs. reading the retrieved '
            f'text</title><desc id="d">{desc}</desc>'
            f'<rect width="{W}" height="{H:.0f}" rx="8" fill="{t["surface"]}"/>')
    return head + "".join(out) + "</svg>"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--results", required=True, help="directory holding the langgraph result folders")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    A, n = audited(Path(a.judge))
    tok = {k: json.loads((Path(a.results) / k / "langgraph_results.json").read_text())["input_tokens_per_q"]
           for k in ("lg_cart", "lg_text", "fix_lg_cart_terse", "lg_text_terse")}
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        (out / f"results-{mode}.svg").write_text(render(t, A, n, tok), encoding="utf-8")
    print(f"wrote results-light.svg and results-dark.svg to {out} (n = {n})")


if __name__ == "__main__":
    main()
