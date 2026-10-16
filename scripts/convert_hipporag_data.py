"""convert_hipporag_data.py — map HippoRAG 2's released evaluation data into this repo's schema.

WHY THIS EXISTS. HippoRAG 2 (arXiv 2502.14802 §4.2) evaluates on 1,000 2Wiki questions over a
6,119-passage corpus, and publishes readers only at Llama-3.3-70B / GPT-4o-mini — no reader under 10B
and no reader-size ablation. Our numbers are on a differently-filtered 500-question draw, so §4.5's
positioning table has to carry "same scale, not the same sample". Converting their release lets
`text_rag.py` run OUR Qwen3-4B reader on THEIR exact sample, turning the reader-scale claim from an
assertion into a measurement.

WHAT IT DOES NOT FIX. Our cartridge chain cannot run on their sample without training cartridges for
their ~2,000 gold articles. This enables the TEXT baselines only.

THEIR SCHEMA (verified against the actual release, 2026-09-19):
  2wikimultihopqa_corpus.json : [ {"title","text"}, ... ]                       -- 6,119 records
  2wikimultihopqa.json        : [ {"_id","type","question","answer","context",
                                   "supporting_facts","evidences",...}, ... ]   -- 1,000 records
`answer` is a plain STRING; there is no `paragraphs` field. `context` is 2Wiki's native
list-of-[title, [sentences]] and `supporting_facts` is list-of-[title, sent_id] — exactly the shapes
`prepare_2wiki.norm_context` / `norm_support_titles` already handle, so those are imported rather than
re-implemented. A MuSiQue-style `paragraphs` layout is also accepted, since sibling files in the same
release use it.

OUR SCHEMA (what prepare_2wiki.py emits and router.py / text_rag.py consume):
  corpus.jsonl    : {"passage_id","title","passage_idx","context","qas"}   ALL passages (retrieval pool)
  articles.jsonl  : same shape, GOLD only                                  (cartridge targets)
  multihop.jsonl  : {"id","question","answer","type","article_slugs"}

Usage:
  python scripts/convert_hipporag_data.py \
      --corpus data/hipporag/raw/2wikimultihopqa_corpus.json \
      --queries data/hipporag/raw/2wikimultihopqa.json \
      --out-dir data/hipporag/2wiki
"""
from __future__ import annotations

import sys
import json
import argparse
import collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# prepare_2wiki has no module-level transformers import (the tokenizer lives inside
# count_tokens_factory), so these are cheap and keep parsing identical across the two corpora.
from prepare_2wiki import slugify, norm_context, norm_support_titles


def load_json(path: Path):
    raw = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [json.loads(l) for l in raw.splitlines() if l.strip()]
    if isinstance(data, dict):
        for k in ("data", "examples", "questions"):
            if k in data and isinstance(data[k], list):
                return data[k]
        raise SystemExit(f"ABORT: {path} is a dict with no list under data/examples/questions "
                         f"(keys: {list(data)[:8]})")
    return data


def gold_titles_and_context(q) -> tuple[list[str], dict[str, str]]:
    """Return (distinct gold titles, title->passage) for either release layout."""
    if q.get("supporting_facts") is not None:            # native 2Wiki
        return norm_support_titles(q["supporting_facts"]), norm_context(q.get("context", []))
    paras = q.get("paragraphs") or []                    # MuSiQue-style sibling files
    titles = [p["title"] for p in paras if p.get("is_supporting")]
    return list(dict.fromkeys(titles)), {p["title"]: p.get("text", "") for p in paras}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="HippoRAG {name}_corpus.json")
    ap.add_argument("--queries", required=True, help="HippoRAG {name}.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-questions", type=int, default=None, help="smoke-test subset")
    ap.add_argument("--id-prefix", default="hippo")
    a = ap.parse_args()

    corpus_raw = load_json(Path(a.corpus))
    queries_raw = load_json(Path(a.queries))
    print(f"loaded {len(corpus_raw)} corpus passages, {len(queries_raw)} queries")

    # ---- corpus: title -> slug, disambiguated exactly as prepare_2wiki does ----
    slug_of: dict[str, str] = {}
    used: set[str] = set()

    def get_slug(title: str) -> str:
        if title in slug_of:
            return slug_of[title]
        base = slug = slugify(title)
        i = 1
        while slug in used:
            slug = f"{base}_{i}"; i += 1
        used.add(slug); slug_of[title] = slug
        return slug

    corpus: dict[str, dict] = {}
    dup_titles = 0
    for r in corpus_raw:
        title = r.get("title")
        if title is None:
            raise SystemExit("ABORT: a corpus record has no 'title'; the release schema changed.")
        text = r.get("text") if r.get("text") is not None else r.get("context", "")
        if title in slug_of:
            dup_titles += 1
            continue
        s = get_slug(title)
        corpus[s] = {"passage_id": s, "title": title, "passage_idx": 0,
                     "context": (text or "").strip(), "qas": []}
    if dup_titles:
        print(f"  note: {dup_titles} duplicate titles collapsed to the first occurrence")

    # ---- queries ----
    questions, gold_slugs = [], set()
    stats = collections.Counter()
    recovered_examples = []
    for i, q in enumerate(queries_raw):
        if a.max_questions is not None and len(questions) >= a.max_questions:
            break
        ans = q.get("answer")
        if isinstance(ans, list):
            if not ans:
                stats["no_answer"] += 1; continue
            ans = ans[0]                       # alias lists: the first entry is the reference
        if not ans:
            stats["no_answer"] += 1; continue

        titles, qctx = gold_titles_and_context(q)
        if not titles:
            stats["no_supporting"] += 1; continue

        slugs, recovered = [], []
        for t in titles:
            if t not in slug_of:
                # Gold title absent from their corpus file. Recover the passage from the question's
                # own context so the question stays answerable; a missing gold would otherwise make it
                # unanswerable by ANY retrieval system and silently depress every row in the table.
                body = (qctx.get(t) or "").strip()
                if not body:
                    stats["gold_unrecoverable"] += 1
                    recovered.append(None)
                    continue
                s = get_slug(t)
                corpus.setdefault(s, {"passage_id": s, "title": t, "passage_idx": 0,
                                      "context": body, "qas": []})
                recovered.append(t)
            s = slug_of[t]
            if s not in slugs:
                slugs.append(s)
        if any(r is None for r in recovered):
            stats["dropped_missing_gold"] += 1
            continue
        if [r for r in recovered if r]:
            stats["gold_recovered_from_context"] += 1
            if len(recovered_examples) < 5:
                recovered_examples.append([r for r in recovered if r][0])

        gold_slugs.update(slugs)
        questions.append({
            "id": q.get("_id") or q.get("id") or f"{a.id_prefix}_{i:05d}",
            "question": q["question"],
            "answer": ans,
            "type": q.get("type", "unknown"),
            "article_slugs": slugs,
        })
        stats[f"gold_titles_{len(slugs)}"] += 1
        stats["kept"] += 1

    if not questions:
        raise SystemExit("ABORT: no questions survived conversion — check the input schema.")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as f:
        for r in corpus.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "articles.jsonl", "w", encoding="utf-8") as f:
        for s, r in corpus.items():
            if s in gold_slugs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "multihop.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    n_empty = sum(1 for r in corpus.values() if not r["context"])
    print(f"\nwrote {out}/")
    print(f"  multihop.jsonl : {len(questions)} questions")
    print(f"  corpus.jsonl   : {len(corpus)} passages   ({n_empty} with empty text)")
    print(f"  articles.jsonl : {len(gold_slugs)} gold passages")
    print(f"  types          : {dict(collections.Counter(q['type'] for q in questions))}")
    print(f"  stats          : {dict(stats)}")
    if recovered_examples:
        print(f"  gold titles recovered from question context: {recovered_examples}")

    # Gates — the assumptions every downstream number rests on.
    if any(not q["article_slugs"] for q in questions):
        raise SystemExit("ABORT: a question ended with no gold slugs.")
    if not gold_slugs <= set(corpus):
        raise SystemExit("ABORT: a gold slug is missing from the corpus — retrieval could never find it.")
    empty_gold = [s for s in gold_slugs if not corpus[s]["context"]]
    if empty_gold:
        raise SystemExit(f"ABORT: {len(empty_gold)} GOLD passages have empty text, e.g. {empty_gold[:3]}")
    print("\ngates OK — every question has >=1 gold slug, every gold slug is in the corpus with text")
    print("NEXT: build_cartridge_index.py --embed-only over corpus.jsonl, then text_rag.py")


if __name__ == "__main__":
    main()
