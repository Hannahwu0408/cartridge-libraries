"""
build_article_corpora.py — group per-passage SQuAD records into per-article corpora.

Deliverable B of Gate 1. The existing pipeline (generate_selfstudy_data → train_squad_cartridges
→ evaluate_squad → benchmark_cost) keys everything on a passages.jsonl whose records are
{passage_id, title, context, qas}. This script emits a file with the SAME record shape but
where each record is a WHOLE ARTICLE: context = all of the article's paragraphs concatenated,
qas = all of the article's questions. That lets one cartridge stand in for a ~2k-token corpus,
which is the setting where cartridge_len ≪ corpus_len makes compression (and a cost win) real.

The article slug is the passage_id with its trailing `_NNN` stripped (e.g. beyonc_003 → beyonc),
matching find_wrong_cartridge_path's title-prefix logic for the cross-article negative control.

Output
------
data/squad/pilot/article_passages.jsonl — one record per article:
    {"passage_id": <slug>, "title": ..., "passage_idx": 0,
     "context": <paragraphs joined by \\n\\n>, "qas": [<all article QAs>]}

Usage
-----
python scripts/build_article_corpora.py \
    --passages-file data/squad/pilot/passages.jsonl \
    --out-file data/squad/pilot/article_passages.jsonl
"""
from __future__ import annotations

import re
import json
import argparse
from pathlib import Path


def article_slug(passage_id: str) -> str:
    """beyonc_003 → beyonc  (strip the trailing _NNN passage index)."""
    return re.sub(r"_\d+$", "", passage_id)


def load_passages(path: Path) -> list[dict]:
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def count_tokens(texts: list[str], model_id: str):
    """Return a list of token counts, or None if the tokenizer can't be loaded."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return [len(tok.encode(t, add_special_tokens=False)) for t in texts]
    except Exception as e:  # no transformers / no network / no cache → fall back
        print(f"  (tokenizer unavailable: {type(e).__name__}; using chars/4 estimate)")
        return None


def main():
    ap = argparse.ArgumentParser(description="Group per-passage SQuAD records into per-article corpora")
    ap.add_argument("--passages-file", default="data/squad/pilot/passages.jsonl")
    ap.add_argument("--out-file", default="data/squad/pilot/article_passages.jsonl")
    ap.add_argument("--model-id", default="Qwen/Qwen3-1.7B",
                    help="Only used to count corpus tokens for the compression-ratio report")
    ap.add_argument("--sweep-lens", default="256,512,1024",
                    help="Cartridge lengths to report the compression ratio against")
    args = ap.parse_args()

    recs = load_passages(Path(args.passages_file))
    print(f"Loaded {len(recs)} per-passage records from {args.passages_file}")

    # Group by article slug, preserving first-seen order.
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in recs:
        slug = article_slug(r["passage_id"])
        if slug not in groups:
            groups[slug] = []
            order.append(slug)
        groups[slug].append(r)

    articles = []
    for slug in order:
        members = groups[slug]
        # Concatenate unique paragraph contexts (preserve order).
        contexts, seen_ctx = [], set()
        for m in members:
            ctx = m["context"]
            if ctx not in seen_ctx:
                seen_ctx.add(ctx)
                contexts.append(ctx)
        corpus = "\n\n".join(contexts)

        # Concatenate all QAs, dedup by normalized question text (first wins).
        qas, seen_q = [], set()
        for m in members:
            for qa in m["qas"]:
                key = qa["question"].strip().lower()
                if key not in seen_q:
                    seen_q.add(key)
                    qas.append(qa)

        articles.append({
            "passage_id": slug,
            "title": members[0]["title"],
            "passage_idx": 0,
            "context": corpus,
            "qas": qas,
            "_n_passages": len(contexts),  # diagnostics (ignored downstream)
        })

    # Token counts for the compression-ratio report.
    tok_counts = count_tokens([a["context"] for a in articles], args.model_id)
    sweep = [int(x) for x in args.sweep_lens.split(",")]

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for a in articles:
            rec = {k: v for k, v in a.items() if not k.startswith("_")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(articles)} article corpora → {out_path}\n")
    print(f"  {'article':<18}{'passages':>9}{'QAs':>6}{'chars':>8}{'tokens':>8}   compression vs " + "/".join(map(str, sweep)))
    print("  " + "-" * 78)
    for i, a in enumerate(articles):
        ntok = tok_counts[i] if tok_counts else len(a["context"]) // 4
        ratios = "  ".join(f"{ntok / L:4.1f}x" for L in sweep)
        tok_str = f"{ntok}" if tok_counts else f"~{ntok}"
        print(f"  {a['passage_id']:<18}{a['_n_passages']:>9}{len(a['qas']):>6}"
              f"{len(a['context']):>8}{tok_str:>8}   {ratios}")

    print(f"\nNext: generate_selfstudy_data.py --passages-file {out_path} "
          f"--out-file data/squad/pilot/article_selfstudy.jsonl")


if __name__ == "__main__":
    main()
