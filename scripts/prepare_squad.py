"""
prepare_squad.py — Download SQuAD 1.1 and export a pilot subset.

Usage
-----
# List available article titles
python scripts/prepare_squad.py --list-articles

# Export pilot passages (first N passages per article)
python scripts/prepare_squad.py \
    --articles "Nikola Tesla,Albert Einstein,Isaac Newton,Immune system,New York City" \
    --passages-per-article 10 \
    --out-dir data/squad/pilot

Output
------
data/squad/pilot/passages.jsonl
    One JSON line per passage:
    {
        "passage_id":    "nikola_tesla_000",   # stable slug_idx key
        "title":         "Nikola Tesla",
        "passage_idx":   0,                    # 0-based index within article
        "context":       "Tesla was born ...", # raw SQuAD paragraph text
        "qas": [                               # QA pairs for this passage
            {
                "question": "When was Nikola Tesla born?",
                "answers": [{"text": "10 July 1856", "answer_start": 17}]
            },
            ...
        ]
    }

data/squad/pilot/articles.json
    Summary: title → number of passages selected.
"""

from __future__ import annotations

import json
import re
import argparse
from pathlib import Path

from datasets import load_dataset


# ── Slug helpers ─────────────────────────────────────────────────────────────────

def slugify(title: str) -> str:
    """Convert article title to a filesystem-safe slug."""
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s


def make_passage_id(title: str, passage_idx: int) -> str:
    return f"{slugify(title)}_{passage_idx:03d}"


# ── SQuAD loading ─────────────────────────────────────────────────────────────────

def load_squad_articles(split: str = "train") -> dict[str, list[dict]]:
    """
    Load SQuAD 1.1 from HuggingFace and return a dict:
        { title: [{"context": str, "qas": [...]}] }

    SQuAD 1.1 HuggingFace format (rajpurkar/squad):
      Each example has: id, title, context, question, answers (dict of lists).
    We re-group them by title and then by context (paragraph).
    """
    print(f"Loading SQuAD 1.1 ({split} split) from HuggingFace ...")
    ds = load_dataset("rajpurkar/squad", split=split)

    # Group by title → context → list of QA dicts
    articles: dict[str, dict[str, list]] = {}
    for ex in ds:
        title = ex["title"]
        context = ex["context"]
        qa = {
            "question": ex["question"],
            "answers": [
                {"text": t, "answer_start": s}
                for t, s in zip(
                    ex["answers"]["text"], ex["answers"]["answer_start"]
                )
            ],
        }
        articles.setdefault(title, {}).setdefault(context, []).append(qa)

    # Convert inner dict to ordered list of paragraphs
    result: dict[str, list[dict]] = {}
    for title, ctx_map in articles.items():
        result[title] = [
            {"context": ctx, "qas": qas} for ctx, qas in ctx_map.items()
        ]

    print(f"Loaded {len(result)} articles, "
          f"{sum(len(ps) for ps in result.values())} total paragraphs.")
    return result


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare a SQuAD 1.1 pilot subset for cartridge training"
    )
    parser.add_argument(
        "--list-articles", action="store_true",
        help="Print all available article titles and exit"
    )
    parser.add_argument(
        "--articles",
        default="Nikola Tesla,Albert Einstein,Isaac Newton,Immune system,New York City",
        help="Comma-separated list of article titles to include in the pilot"
    )
    parser.add_argument(
        "--passages-per-article", type=int, default=10,
        help="Max passages to select per article (default: 10)"
    )
    parser.add_argument(
        "--min-qas", type=int, default=2,
        help="Skip passages with fewer than this many QA pairs (default: 2)"
    )
    parser.add_argument(
        "--out-dir", default="data/squad/pilot",
        help="Output directory for passages.jsonl and articles.json"
    )
    parser.add_argument(
        "--squad-split", default="train",
        help="SQuAD split to use: train or validation (default: train)"
    )
    args = parser.parse_args()

    articles = load_squad_articles(args.squad_split)

    # Build a normalized lookup: spaces→underscores and underscores→spaces both map to
    # the actual stored key. SQuAD stores titles with underscores (Wikipedia URL format).
    normalized: dict[str, str] = {}
    for title in articles:
        normalized[title] = title                          # exact match
        normalized[title.replace("_", " ")] = title       # underscore → space
        normalized[title.replace(" ", "_")] = title       # space → underscore

    if args.list_articles:
        print(f"\n{'Article title':<50}  {'Paragraphs':>10}  {'QA pairs':>8}")
        print("-" * 72)
        for title in sorted(articles):
            n_para = len(articles[title])
            n_qa = sum(len(p["qas"]) for p in articles[title])
            print(f"{title:<50}  {n_para:>10}  {n_qa:>8}")
        return

    # Resolve user-supplied titles via normalized lookup
    raw_wanted = [t.strip() for t in args.articles.split(",")]
    wanted: list[str] = []          # resolved canonical titles
    missing: list[str] = []
    for t in raw_wanted:
        if t in normalized:
            wanted.append(normalized[t])
        else:
            missing.append(t)

    if missing:
        print(f"\nWARNING: the following articles were not found in SQuAD 1.1:")
        for t in missing:
            print(f"  '{t}'")
        print("Tip: SQuAD titles use underscores (e.g. 'Kanye_West', 'New_York_City').")
        print("Run --list-articles to see all available titles.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    passages_out = []
    summary: dict[str, int] = {}

    for title in wanted:
        if title not in articles:
            continue

        paragraphs = articles[title]
        selected = 0
        for para in paragraphs:
            if selected >= args.passages_per_article:
                break
            if len(para["qas"]) < args.min_qas:
                continue

            passage_id = make_passage_id(title, selected)
            passages_out.append(
                {
                    "passage_id": passage_id,
                    "title": title,
                    "passage_idx": selected,
                    "context": para["context"],
                    "qas": para["qas"],
                }
            )
            selected += 1

        summary[title] = selected
        print(f"  {title}: {selected} passages selected "
              f"(from {len(paragraphs)} available)")

    # Write passages.jsonl
    passages_path = out_dir / "passages.jsonl"
    with open(passages_path, "w", encoding="utf-8") as f:
        for p in passages_out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Write articles.json (summary)
    articles_path = out_dir / "articles.json"
    articles_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total_passages = sum(summary.values())
    total_qas = sum(
        len(p["qas"]) for p in passages_out
    )
    print(f"\nWrote {total_passages} passages, {total_qas} QA pairs → {passages_path}")
    print(f"Article summary → {articles_path}")
    print(f"\nNext: python scripts/train_squad_cartridges.py --passages-file {passages_path}")


if __name__ == "__main__":
    main()
