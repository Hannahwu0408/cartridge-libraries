"""make_poster_figures.py — the two poster figures for the results section.

FIG 1  accuracy by metric, four systems, ALL under terse decoding (--short-answer).
       One decoding for every row is the whole point: under verbose decoding strict EM is
       0.000 for EVERY system including the teacher, which measures the metric rather than
       the system. The audited column is the blind LLM judge (grades the answer ASSERTED,
       not whether the gold string appears).
FIG 2  the self-study scaling curve, audited + dataset-adjusted (n=96).

Palette: dataviz reference categorical slots 1-4, validated (CVD worst adjacent dE 9.1
protan / 22.9 normal; the sub-3:1 contrast WARN is discharged by direct-labelling every bar).

  python scripts/make_poster_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
SURF = "#ffffff"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 15,
    "axes.edgecolor": INK3, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})

# ── FIG 1 ────────────────────────────────────────────────────────────────────────────────
metrics = ["Exact Match", "token F1", "contains-EM", "LLM-audited"]
systems = [
    ("Cartridge chain (ours)", [0.400, 0.557, 0.540, 0.710], S1),
    ("IRCoT (text, same controller)", [0.410, 0.521, 0.500, 0.670], S2),
    ("4B teacher (gold text)", [0.460, 0.566, 0.560, 0.730], S3),
    ("Claude Opus + gold (ceiling)", [0.660, 0.761, 0.780, 0.940], S4),
]
fig, ax = plt.subplots(figsize=(12.5, 6.4))
x = np.arange(len(metrics)); w = 0.20
for i, (name, vals, c) in enumerate(systems):
    off = (i - 1.5) * w
    ax.bar(x + off, vals, w * 0.92, label=name, color=c, zorder=3)   # 2px-equivalent gap
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.018, f"{v:.2f}", ha="center", va="bottom",
                fontsize=11.5, color=INK2, zorder=4)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.04); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylabel("accuracy")
ax.set_title("Cartridge chain vs. text baselines and teacher\non 100 2WikiMultihopQA bridge questions",
             fontsize=22, color=INK, pad=20, loc="center", linespacing=1.35)
ax.grid(axis="y", color="#e6e5e1", lw=0.9, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.10), fontsize=13)
fig.tight_layout()
fig.savefig("figures/fig1_accuracy_by_metric.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/fig1_accuracy_by_metric.pdf", bbox_inches="tight")
print("wrote figures/fig1_accuracy_by_metric.{png,pdf}")

# ── FIG 2 ─────────────────────────────────────────────────────────────────────
pp = np.array([1000, 2000, 4000]); acc = np.array([0.667, 0.760, 0.740])
fig, ax = plt.subplots(figsize=(10.2, 6.6))

ax.axhline(0.969, color=S4, lw=2, ls=(0, (6, 4)), zorder=2,
           label="Claude Opus + gold articles (ceiling)  0.969")
ax.axhline(0.708, color=S3, lw=2, ls=(0, (6, 4)), zorder=2,
           label="4B teacher, both gold articles as text  0.708")

ax.plot(pp, acc, "-o", color=S1, lw=2.6, ms=12, zorder=4,
        markeredgecolor=SURF, markeredgewidth=2,
        label="Cartridge chain (ours)")
for p_, a_, dy, va in zip(pp, acc, (-0.028, 0.024, 0.024), ("top", "bottom", "bottom")):
    ax.text(p_, a_ + dy, f"{a_:.3f}", ha="center", va=va, fontsize=15, color=INK)

ax.text(1414, 0.632, "+9 questions\np = 0.012", ha="center", va="bottom",
        fontsize=13, color=INK)
ax.text(2828, 0.786, "\u22121 question\np = 1.00   null", ha="center", va="bottom",
        fontsize=13, color=INK)

ax.set_xscale("log", base=2); ax.set_xticks(pp)
ax.set_xticklabels([f"{v:,}" for v in pp])
ax.set_xlim(900, 4900); ax.set_ylim(0.60, 1.03)
ax.set_yticks(np.arange(0.6, 1.01, 0.1))
ax.set_xlabel("self-study conversations per passage", labelpad=10)
ax.set_ylabel("audited accuracy  (dataset-adjusted, n = 96)", labelpad=10)
ax.set_title("Cartridge training data vs. multi-hop accuracy\n"
             "returns saturate at 2,000 conversations per passage",
             fontsize=18, color=INK, pad=20, loc="center", linespacing=1.35)
_h, _l = ax.get_legend_handles_labels()          # main series first, references after
ax.legend(_h[::-1], _l[::-1], frameon=False, loc="upper left",
          bbox_to_anchor=(0, -0.21), fontsize=13, labelcolor=INK,
          handlelength=2.6, markerscale=0.8)
ax.grid(axis="y", color="#e6e5e1", lw=0.9, zorder=0); ax.set_axisbelow(True)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig("figures/fig2_scaling_curve.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/fig2_scaling_curve.pdf", bbox_inches="tight")
print("wrote figures/fig2_scaling_curve.{png,pdf}")


# ── FIG 3  context footprint (memory) ────────────────────────────────────────────
# Per hop, medians over 40 timed queries (benchmark_cost.py, Qwen3-4B, 2Wiki articles).
# "Input tokens" = what the reader must process at query time; "KV positions" = what it
# must hold in cache. A cartridge contributes ZERO input tokens: its document is already
# a trained KV prefix, so it is billed as positions, not as prompt.
fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2))
panels = [
    ("Input tokens processed per hop\n8.8\u00d7 fewer",
     ["Cartridge\n(ours)", "Text,\narticle prefilled"], [32, 283], [None, 649]),
    ("KV cache positions held per hop\n1.7\u00d7 smaller",
     ["Cartridge\n(ours)", "Text,\narticle prefilled"], [192, 331], [None, 697]),
]
for ax_, (title, labels, med, p90) in zip(axes, panels):
    xs = np.arange(2)
    ax_.bar(xs, med, 0.52, color=[S1, S2], zorder=3)
    for x_, m_, p_ in zip(xs, med, p90):
        ax_.text(x_, m_ + max(med) * 0.03, f"{m_:,}", ha="center", va="bottom",
                 fontsize=16, color=INK)
        if p_:                                   # whisker on the bar's right edge, clear of the label
            xw = x_ + 0.26
            ax_.plot([xw, xw], [m_, p_], color=INK3, lw=1.6, zorder=4)
            ax_.plot([xw - 0.06, xw + 0.06], [p_, p_], color=INK3, lw=1.6, zorder=4)
            ax_.text(xw + 0.10, p_, f"p90 {p_:,}", ha="left", va="center",
                     fontsize=12.5, color=INK)
    ax_.set_xticks(xs); ax_.set_xticklabels(labels, fontsize=14)
    ax_.set_xlim(-0.55, 1.75)
    ax_.set_ylim(0, max(max(med), max(v for v in p90 if v)) * 1.22)
    ax_.set_title(title, fontsize=15, color=INK, pad=14, linespacing=1.4)
    ax_.grid(axis="y", color="#e6e5e1", lw=0.9, zorder=0); ax_.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax_.spines[sp].set_visible(False)
fig.suptitle("Serving cost per hop: a cartridge replaces the document with a fixed KV prefix",
             fontsize=22, color=INK, y=1.04)
fig.tight_layout()
fig.savefig("figures/fig3_context_footprint.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/fig3_context_footprint.pdf", bbox_inches="tight")
print("wrote figures/fig3_context_footprint.{png,pdf}")

# ── FIG 4  prefill latency ───────────────────────────────────────────────────────
# The claim is length-INVARIANCE, not a wall-clock win: per-token decode is at parity
# (~15.2 ms/tok everywhere), so total latency is dominated by generation length.
fig, ax = plt.subplots(figsize=(9.4, 6.2))
conds = ["Cartridge\n(ours)", "Text,\narticle prefilled", "Text,\nprefix-cached"]
med = [17.6, 25.4, 17.9]; p90 = [None, 44.7, None]
xs = np.arange(3)
ax.bar(xs, med, 0.5, color=[S1, S2, S3], zorder=3)
for x_, m_, p_ in zip(xs, med, p90):
    ax.text(x_, m_ + 1.0, f"{m_} ms", ha="center", va="bottom", fontsize=15.5, color=INK)
    if p_:
        xw = x_ + 0.25
        ax.plot([xw, xw], [m_, p_], color=INK3, lw=1.6, zorder=4)
        ax.plot([xw - 0.055, xw + 0.055], [p_, p_], color=INK3, lw=1.6, zorder=4)
        ax.text(xw + 0.09, p_, f"p90 {p_} ms", ha="left", va="center", fontsize=13, color=INK)
ax.set_xticks(xs); ax.set_xticklabels(conds, fontsize=14)
ax.set_ylabel("prefill latency per hop (ms)", labelpad=10)
ax.set_ylim(0, 52)
ax.set_title("Prefill cost is flat for a cartridge and grows with document length for text",
             fontsize=20, color=INK, pad=18, loc="center")
ax.text(0.5, -0.30, "Per-token decode is at parity (~15.2 ms/token) in all three conditions \u2014\n"
        "the claim is bounded, length-invariant context, not a wall-clock speedup.",
        transform=ax.transAxes, ha="center", fontsize=12.5, color=INK)
ax.grid(axis="y", color="#e6e5e1", lw=0.9, zorder=0); ax.set_axisbelow(True)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig("figures/fig4_prefill_latency.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/fig4_prefill_latency.pdf", bbox_inches="tight")
print("wrote figures/fig4_prefill_latency.{png,pdf}")
