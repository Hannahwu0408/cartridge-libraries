"""
build_cartridge_index.py — build the router's cartridge index (Phase 8, Stage 1).

For every article (source section) that has a trained cartridge, embed the section text with a
FROZEN sentence embedder and record one entry in a single index.json:

    { id, embedding, title, passage_id, version, tensor_path }

The router (router.py) loads only this lightweight index, does kNN over the embeddings, and
lazy-loads just the retrieved cartridge tensors to the GPU for composition — retrieval never
touches the heavy tensors. The embedder is FROZEN (a stable coordinate system for continual
learning): record its name so router.py uses the exact same one.

Usage
-----
python scripts/build_cartridge_index.py \
    --articles-file data/synth/articles_mm.jsonl \
    --cartridge-dir cartridges/synth_ca_gnorm --cartridge-len 64 \
    --embedder sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="Build the router's cartridge index.json")
    ap.add_argument("--articles-file", required=True,
                    help="jsonl of source sections: {passage_id, title, context, ...}")
    ap.add_argument("--cartridge-dir", required=True,
                    help="dir holding <passage_id>/cartridge_<len>.pt per article")
    ap.add_argument("--cartridge-len", type=int, default=64)
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="FROZEN sentence-transformer; recorded so router.py matches it")
    ap.add_argument("--query-instruction", default="",
                    help="prefix router.py must prepend to QUERIES (BGE/E5 asymmetric); recorded "
                         "in the index so it can't drift. BGE: 'Represent this sentence for "
                         "searching relevant passages: '  E5: 'query: '")
    ap.add_argument("--passage-instruction", default="",
                    help="prefix prepended to PASSAGES here. E5: 'passage: '")
    ap.add_argument("--out-file", default=None, help="default: <cartridge-dir>/index.json")
    ap.add_argument("--version", type=int, default=1,
                    help="version stamped on every cartridge in this build (continual versioning: a "
                         "fresher rebuild uses a higher number so version-aware routing can prefer it)")
    ap.add_argument("--embed-only", action="store_true",
                    help="index EVERY passage (embeddings for retrieval) even if it has no cartridge — for the "
                         "full retrieval corpus of the scaled benchmark (text-RAG uses embeddings only; the "
                         "cartridge chain honestly fails on un-cartridged distractors).")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    arts = load_jsonl(args.articles_file)
    cart_dir = Path(args.cartridge_dir)
    print(f"Loading frozen embedder: {args.embedder}")
    embedder = SentenceTransformer(args.embedder)

    entries, missing = [], 0
    for a in arts:
        pid = a["passage_id"]
        tpath = cart_dir / pid / f"cartridge_{args.cartridge_len}.pt"
        if not tpath.exists():
            missing += 1
            if not args.embed_only:            # default: only index passages that HAVE a cartridge
                continue
            # --embed-only: index EVERY passage for retrieval (full corpus). tensor_path is set regardless;
            # text-RAG ignores it, and the cartridge chain gets None → honest fail on un-cartridged distractors.
        emb = embedder.encode(args.passage_instruction + a["context"], normalize_embeddings=True)
        entries.append({
            "id": pid,
            "embedding": [float(x) for x in emb],
            "title": a.get("title", pid),
            "passage_id": pid,
            "version": args.version,            # §continual versioning (stamp; router can prefer freshest)
            "tensor_path": str(tpath),
        })

    dim = len(entries[0]["embedding"]) if entries else 0
    out = Path(args.out_file or (cart_dir / "index.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "embedder": args.embedder,
        "query_instruction": args.query_instruction,
        "passage_instruction": args.passage_instruction,
        "cartridge_len": args.cartridge_len,
        "dim": dim,
        "cartridges": entries,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}: {len(entries)} cartridges (dim={dim})"
          + (f"; {missing} articles had no cartridge (skipped)" if missing else ""))


if __name__ == "__main__":
    main()
