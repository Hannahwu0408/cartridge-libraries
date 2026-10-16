"""check_coverage.py — do a self-study arm's QUESTIONS cover the article's facts? (Step 0e)

THE OTHER HALF OF THE 2x2 BAKEOFF. `claude_datacheck.py` measures whether the ANSWERS are correct
(the M_A axis). This measures whether the QUESTIONS reach the article's content at all (the M_Q axis
— CAS's actual decoupled-generator lever, +6.8 pts in their Table 6). The two are expected to move in
OPPOSITE directions: a stronger question model asks harder/broader things (coverage UP, contradiction
rate possibly UP too), a stronger answer model gets them right (contradiction DOWN). Reading only one
metric would credit or blame the wrong role.

Nothing here calls an API or a GPU — it is regex + set membership over the generated jsonl.

MATCHED BUDGET IS NOT OPTIONAL: coverage grows monotonically with convs/passage (more questions ⇒
more facts touched), so cells are compared only after subsampling every cell to the SAME number of
convs per article (default: auto = the min available across cells). Comparing a 1000/pp pool against
a fresh 40/pp sample would measure the budget, not the model.

Metrics per cell (per article, then averaged):
  cov_q      share of the article's salient terms (capitalised names / numbers) that appear in at
             least one QUESTION            <- the M_Q metric: what did the generator think to ask about
  cov_qa     same, but a term counts if it appears in a question OR an answer
             (Q+A is what the cartridge is actually distilled on, so this is the training-coverage view)
  eval_ans   share of this article's multihop GOLD ANSWERS that appear anywhere in its self-study
             (eval-anchored: is the fact the benchmark asks for even present in the training data?)
  eval_brg   share of the bridge entities (the co-gold article's title, for questions this article
             starts) that appear — the hop-1 link the chain controller must decode
  uniq_q     distinct-question rate (dedup health)  ·  q_words / a_words  mean lengths

`salient_terms` is imported from eval_generality so this coverage number is defined identically to
the one already reported in the generality probe (§5.4c); `containment` is the repo's normalised
gold-in-text metric, so eval_ans/eval_brg use the same matching rule as every accuracy table.

Usage (Mac, after rsyncing the four bakeoff cells):
  python scripts/check_coverage.py \
      --cell q4a4:data/2wiki_v2/ss_fid30_q4a4.jsonl \
      --cell q8a4:data/2wiki_v2/ss_fid30_q8a4.jsonl \
      --cell q4a8:data/2wiki_v2/ss_fid30_q4a8.jsonl \
      --cell q8a8:data/2wiki_v2/ss_fid30_q8a8.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_generality import salient_terms          # noqa: E402  (same definition as §5.4c probe)
from evaluate_squad import containment             # noqa: E402  (same matching rule as accuracy)


def load_jsonl(p):
    """Tolerant reader: rsyncs between machines on this project have repeatedly truncated the final
    line, and a hard JSONDecodeError there would kill a 4-cell comparison over one broken record.
    Skip unparseable lines but REPORT them — a large count means a partial transfer, and coverage
    computed on a partial cell would be silently understated."""
    rows, bad = [], 0
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"  ⚠️  {p}: skipped {bad} unparseable line(s) "
              f"({bad/(len(rows)+bad):.1%} of file) — likely a truncated transfer; re-rsync if >1")
    return rows


def frac(hits, total):
    return hits / total if total else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True, metavar="NAME:PATH",
                    help="repeatable: a bakeoff cell's self-study jsonl (e.g. q8a4:data/.../x.jsonl)")
    ap.add_argument("--articles-file", default="data/2wiki_v2/articles.jsonl")
    ap.add_argument("--slugs-file", default="data/2wiki_v2/fidelity30_slugs.txt")
    ap.add_argument("--multihop-file", default="data/2wiki_v2/multihop.jsonl",
                    help="for the eval-anchored columns; skipped if the file is absent")
    ap.add_argument("--k", type=int, default=20,
                    help="salient terms per article (eval_generality's own probe uses 12; more terms "
                         "= a stricter coverage test, which is what we want here)")
    ap.add_argument("--per-article", type=int, default=0,
                    help="convs sampled per article per cell; 0 = AUTO (min available across cells)")
    ap.add_argument("--samples", type=int, default=3, help="example questions printed per cell")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    arts = {x["passage_id"]: x for x in load_jsonl(a.articles_file)}
    slugs = [s.strip() for s in Path(a.slugs_file).read_text().splitlines() if s.strip()] \
        if Path(a.slugs_file).exists() else list(arts)
    slugs = [s for s in slugs if s in arts]

    cells = {}
    for spec in a.cell:
        name, _, path = spec.partition(":")
        if not path:
            sys.exit(f"ABORT: --cell wants NAME:PATH, got {spec!r}")
        by_art = defaultdict(list)
        for c in load_jsonl(path):
            if c["passage_id"] in set(slugs):
                by_art[c["passage_id"]].append(c)
        cells[name] = by_art
        print(f"cell {name:<8} {sum(len(v) for v in by_art.values()):>6} convs "
              f"over {len(by_art)}/{len(slugs)} articles  ({path})")

    # ── matched budget: same articles, same convs/article, in every cell ──────────────────
    common = [s for s in slugs if all(cell.get(s) for cell in cells.values())]
    dropped = [s for s in slugs if s not in common]
    if dropped:
        print(f"\n⚠️  {len(dropped)} articles absent from ≥1 cell → excluded from ALL cells "
              f"(a cell must not be credited for articles another cell never generated): "
              f"{dropped[:4]}{'...' if len(dropped) > 4 else ''}")
    if not common:
        sys.exit("ABORT: no article is present in every cell — nothing comparable to measure.")

    n_per = {}
    for s in common:
        avail = min(len(cells[c][s]) for c in cells)
        n_per[s] = min(a.per_article, avail) if a.per_article else avail
    print(f"\nmatched budget: {len(common)} articles · "
          f"{min(n_per.values())}-{max(n_per.values())} convs/article "
          f"(mean {sum(n_per.values())/len(n_per):.0f}){' [--per-article cap]' if a.per_article else ' [auto=min across cells]'}")

    mh = load_jsonl(a.multihop_file) if Path(a.multihop_file).exists() else []
    # answering article = LAST gold slug (the one holding the answer); bridge = the co-gold's title
    ans_targets, brg_targets = defaultdict(list), defaultdict(list)
    for q in mh:
        sl = q.get("article_slugs") or []
        if len(sl) >= 2:
            ans_targets[sl[-1]].append(q["answer"])
            brg_targets[sl[0]].append(arts.get(sl[-1], {}).get("title", sl[-1]))

    rows = {}
    for name, by_art in cells.items():
        per = defaultdict(list)
        for s in common:
            # fresh per (cell, article) so the subsample is deterministic and order-independent
            convs = random.Random(f"{a.seed}:{s}").sample(by_art[s], n_per[s]) \
                if n_per[s] < len(by_art[s]) else list(by_art[s])
            qs = [c["question"] for c in convs]
            qa = [c["question"] + " " + (c.get("answer") or "") for c in convs]
            q_blob, qa_blob = " ".join(qs).lower(), " ".join(qa).lower()

            terms = salient_terms(arts[s]["context"], k=a.k)
            per["cov_q"].append(frac(sum(t.lower() in q_blob for t in terms), len(terms)))
            per["cov_qa"].append(frac(sum(t.lower() in qa_blob for t in terms), len(terms)))
            per["uniq_q"].append(frac(len({q.lower() for q in qs}), len(qs)))
            per["q_words"].append(sum(len(q.split()) for q in qs) / max(1, len(qs)))
            per["a_words"].append(sum(len((c.get("answer") or "").split()) for c in convs)
                                  / max(1, len(convs)))
            if ans_targets.get(s):
                per["eval_ans"].append(frac(sum(containment(qa_blob, g) for g in ans_targets[s]),
                                            len(ans_targets[s])))
            if brg_targets.get(s):
                per["eval_brg"].append(frac(sum(containment(qa_blob, b) for b in brg_targets[s]),
                                            len(brg_targets[s])))
        rows[name] = {k: sum(v) / len(v) for k, v in per.items() if v}

    cols = ["cov_q", "cov_qa", "eval_ans", "eval_brg", "uniq_q", "q_words", "a_words"]
    print("\n" + "=" * 74)
    print("QUESTION-COVERAGE at matched budget (higher = the generator asked about more of the doc)")
    print(f"  {'cell':<10}" + "".join(f"{c:>10}" for c in cols))
    for name, m in rows.items():
        print(f"  {name:<10}" + "".join(
            f"{m[c]:>10.2f}" if c in m else f"{'-':>10}" for c in cols))
    n_ans = sum(1 for s in common if ans_targets.get(s))
    n_brg = sum(1 for s in common if brg_targets.get(s))
    print(f"  (eval_ans over {n_ans} articles with a gold answer · eval_brg over {n_brg} with a bridge)")
    print("\n  READ: cov_q is the M_Q metric — compare q4* vs q8* rows. If cov_q rises with the 8B\n"
          "  question model, the coverage lever is real and carries even if that cell's contradiction\n"
          "  rate is worse (pair it with a stronger answerer). If cov_q is flat, M_Q is not the\n"
          "  binding constraint and no --question-model work is warranted.")

    if a.samples:
        print("\n=== sample questions (first common article) ===")
        s0 = common[0]
        for name, by_art in cells.items():
            print(f"\n  [{name}] {s0}")
            for c in by_art[s0][: a.samples]:
                print(f"    Q: {c['question'][:110]}")

    out = Path("results/datacheck/coverage.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"articles": common, "n_per_article": n_per,
                               "k": a.k, "cells": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
