"""
prepare_2wiki.py — turn 2WikiMultihopQA into per-article corpora + multi-hop questions
for the composition test (does composing 2 cartridges beat either alone?).

2WikiMultihopQA example schema (HotpotQA-style, from the official `dev.json`):
  _id, question, answer, type ∈ {comparison, inference, compositional, bridge_comparison},
  context: [[title, [sentences]], ...]   (candidate paragraphs; ~10)
  supporting_facts: [[title, sent_id], ...]   (which sentences/titles support the answer)
  entity_ids, evidences (ignored here)

We keep questions whose supporting_facts span EXACTLY 2 distinct gold titles (the two
corpora to compose), and where both gold paragraphs are long enough to fill a common
cartridge_len (composition requires equal lengths; init_from_passage shrinks the cartridge
to passage_len for short passages, so we drop paragraphs below --min-tokens).

Get dev.json from the official release (GitHub: Alab-NII/2wikimultihop → data.zip) and pass
its path via --dev-json.

Output
------
data/2wiki/articles.jsonl   — one record per UNIQUE gold article (passages.jsonl shape):
    {"passage_id": <title-slug>, "title", "passage_idx": 0, "context": <sentences joined>, "qas": []}
data/2wiki/multihop.jsonl   — one record per kept question:
    {"id", "question", "answer", "type", "article_slugs": [slugA, slugB]}

Usage
-----
python scripts/prepare_2wiki.py --dev-json data/2wiki/dev.json \
    --types comparison,compositional,inference --min-tokens 128 --max-questions 40 \
    --out-dir data/2wiki
"""
from __future__ import annotations

import re
import json
import argparse
from pathlib import Path


def slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return s or "article"


def load_examples(path: Path) -> list[dict]:
    """Accept either a JSON array (official dev.json) or JSONL (HF `Dataset.to_json`)."""
    raw = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):           # some dumps wrap as {"data": [...]}
            data = data.get("data", [data])
        return data
    except json.JSONDecodeError:
        return [json.loads(l) for l in raw.splitlines() if l.strip()]


def _maybe_json(x):
    """Parquet/HF dumps store context & supporting_facts as JSON strings — decode them."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return x
    return x


def norm_context(ctx) -> dict[str, str]:
    """title -> joined paragraph, for list-of-[title,sents], dict-of-lists, or JSON string."""
    ctx = _maybe_json(ctx)
    if isinstance(ctx, dict):
        titles = ctx.get("title") or ctx.get("titles") or []
        sents = ctx.get("sentences") or ctx.get("content") or ctx.get("sents") or []
        return {t: " ".join(s).strip() for t, s in zip(titles, sents)}
    return {t: " ".join(s).strip() for t, s in ctx}


def norm_support_titles(sf) -> list[str]:
    """distinct supporting titles, for list-of-[title,sent_id], dict-of-lists, or JSON string."""
    sf = _maybe_json(sf)
    if isinstance(sf, dict):
        return list(dict.fromkeys(sf.get("title", [])))
    return list(dict.fromkeys(t for t, _ in sf))


def count_tokens_factory(model_id: str):
    """Return a fn text->token_count; real tokenizer if available, else chars/4 estimate."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return lambda t: len(tok.encode(t, add_special_tokens=False))
    except Exception as e:
        print(f"  (tokenizer unavailable: {type(e).__name__}; using chars/4 estimate)")
        return lambda t: len(t) // 4


def main():
    ap = argparse.ArgumentParser(description="Prepare 2WikiMultihopQA for the composition test")
    ap.add_argument("--dev-json", required=True, help="Path to 2Wiki dev.json (official release)")
    ap.add_argument("--out-dir", default="data/2wiki")
    ap.add_argument("--types", default="comparison,compositional,inference",
                    help="Comma-separated question types to keep (bridge_comparison has 4 articles)")
    ap.add_argument("--min-tokens", type=int, default=128,
                    help="Drop a question unless BOTH gold paragraphs have >= this many tokens "
                         "(must be >= the cartridge_len you'll train at)")
    ap.add_argument("--max-questions", type=int, default=40, help="Cap kept questions (pilot small)")
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B", help="Tokenizer for the length filter")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    keep_types = {t.strip() for t in args.types.split(",")}
    n_tokens = count_tokens_factory(args.model_id)

    data = load_examples(Path(args.dev_json))
    print(f"Loaded {len(data)} examples from {args.dev_json}")

    slug_of: dict[str, str] = {}     # title -> slug (disambiguated)
    used_slugs: set[str] = set()
    def get_slug(title: str) -> str:
        if title in slug_of:
            return slug_of[title]
        base = slugify(title)
        slug = base
        i = 1
        while slug in used_slugs:
            slug = f"{base}_{i}"; i += 1
        used_slugs.add(slug); slug_of[title] = slug
        return slug

    articles: dict[str, dict] = {}   # slug -> GOLD article record (cartridge-training targets)
    corpus: dict[str, dict] = {}     # slug -> ALL candidate passages (gold + distractor) = retrieval corpus
    questions: list[dict] = []
    stats = {"wrong_type": 0, "not_two_titles": 0, "title_missing_ctx": 0, "too_short": 0, "kept": 0}

    def rec(t, passage):
        return {"passage_id": get_slug(t), "title": t, "passage_idx": 0, "context": passage, "qas": []}

    for ex in data:
        if ex.get("type") not in keep_types:
            stats["wrong_type"] += 1
            continue
        gold_titles = norm_support_titles(ex.get("supporting_facts", []))
        if len(gold_titles) != 2:
            stats["not_two_titles"] += 1
            continue
        ctx = norm_context(ex.get("context", []))
        if any(t not in ctx or not ctx[t] for t in gold_titles):
            stats["title_missing_ctx"] += 1
            continue
        if any(n_tokens(ctx[t]) < args.min_tokens for t in gold_titles):
            stats["too_short"] += 1
            continue

        slugs = []
        for t in gold_titles:
            r = rec(t, ctx[t]); slugs.append(r["passage_id"])
            articles.setdefault(r["passage_id"], r)
        # full candidate corpus = EVERY passage in this kept question's context (supporting + distractor)
        for t, passage in ctx.items():
            if passage:
                r = rec(t, passage); corpus.setdefault(r["passage_id"], r)
        questions.append({
            "id": ex.get("_id"), "question": ex["question"], "answer": ex["answer"],
            "type": ex["type"], "article_slugs": slugs,
        })
        stats["kept"] += 1
        if len(questions) >= args.max_questions:
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_path = out_dir / "articles.jsonl"        # GOLD only → cartridge-training targets
    corpus_path = out_dir / "corpus.jsonl"       # full candidate corpus → retrieval embedding index
    mh_path = out_dir / "multihop.jsonl"
    for path, recs in [(art_path, articles.values()), (corpus_path, corpus.values())]:
        with open(path, "w", encoding="utf-8") as f:
            for a in recs:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(mh_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nFilter stats: {stats}")
    print(f"Wrote {len(questions)} questions ({mh_path})")
    print(f"  GOLD (cartridge targets): {len(articles)} articles  ({art_path})")
    print(f"  CORPUS (retrieval pool):  {len(corpus)} passages  ({corpus_path})  "
          f"[{len(corpus) - len(articles)} distractors]")
    by_type: dict[str, int] = {}
    for q in questions:
        by_type[q["type"]] = by_type.get(q["type"], 0) + 1
    print(f"  by type: {by_type}")
    print(f"\nNext: generate_selfstudy_data.py --passages-file {art_path} --model-id {args.model_id} "
          f"--per-passage 60 --out-file {out_dir}/selfstudy.jsonl")


if __name__ == "__main__":
    main()
