"""rewrite_decoys_matched.py — make every homonym decoy a TEMPLATE-MATCHED twin of its target.

WHY. In the first controlled run, per-hop retrieval was two-regime: ~0.97 at intermediate hops and
~0.51 at the final hop. Cause: `build_synth_chain.py` renders EVERY decoy with `answer_article`, so a
decoy is a true twin only when its target is also an answer article — i.e. the last hop. At an
intermediate hop the real article is a `mentor_article` (different template, and it repeats the entity
name ~4x vs 2x), so the retriever separates them easily. The observed depth-decay was therefore driven
by one hard hop rather than by compounding across equally-hard hops.

FIX. Re-render each decoy from the SAME template as the entity it shadows: mentor-template decoys for
intermediate targets (pointing at a dead-end mentor that has no article), answer-template decoys for
final targets (already correct — left untouched). Every hop then has an identical-looking twin and
per-hop p should fall to ~0.5 everywhere, so end-to-end must decay as p^N by compounding.

COST. Decoys are `--embed-only` (they carry NO cartridge), so this changes nothing that was trained:
rewrite -> re-index -> re-eval, ~1 h, zero training. Chain slugs are preserved exactly, so the 360
trained cartridges stay valid. (This is why we rewrite the EXISTING corpus rather than regenerating
from seed — regeneration would shift the RNG stream and rename every chain entity.)

Usage:
  python scripts/rewrite_decoys_matched.py --data-dir data/synth_chain2 \
      --out-file data/synth_chain2/corpus_matched.jsonl
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

from build_synth_chain import Gen, CRAFT, mentor_article, answer_article


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-file", default=None)
    ap.add_argument("--seed", type=int, default=1234, help="separate stream from the original build")
    a = ap.parse_args()
    d = Path(a.data_dir)
    out_file = Path(a.out_file) if a.out_file else d / "corpus_matched.jsonl"

    chain = load_jsonl(d / "articles.jsonl")          # chain entities (these HAVE cartridges)
    corpus = load_jsonl(d / "corpus.jsonl")           # chain entities + homonym decoys
    mh = load_jsonl(d / "multihop.jsonl")
    chain_slugs = {r["passage_id"] for r in chain}

    # Role of each chain entity, from the chains themselves: E_1..E_{N-1} are mentor-link articles
    # (intermediate hop targets); E_N is the birthplace article (the final hop target). E_0 is never a
    # hop target and never gets a decoy.
    role: dict[str, str] = {}
    for q in mh:
        cs = q["chain_slugs"]
        for s in cs[1:-1]:
            role[s] = "intermediate"
        role[cs[-1]] = "final"
    title_role = {r["title"]: role[r["passage_id"]]
                  for r in chain if r["passage_id"] in role}

    # Name generator for the decoys' dead-end mentors / contemporaries. Pre-seed `used_names` with every
    # real title so a decoy can never accidentally name a real chain entity (which would forge a link).
    g = Gen(a.seed)
    g.used_names.update(r["title"] for r in corpus)

    n_int = n_fin = 0
    rows = []
    for r in corpus:
        if r["passage_id"] in chain_slugs:            # a real chain entity — never touch
            rows.append(r); continue
        rl = title_role.get(r["title"])
        if rl == "intermediate":                      # re-render as a mentor-template twin
            r = dict(r)
            r["context"] = mentor_article(r["title"], g.name(), g.r.choice(CRAFT),
                                          [g.name(), g.name()])
            n_int += 1
        else:                                         # final-hop decoy is already template-matched
            n_fin += 1
        rows.append(r)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_decoy = len(corpus) - len(chain_slugs)
    print(f"wrote {out_file}: {len(rows)} records "
          f"({len(chain_slugs)} chain entities untouched, {n_decoy} decoys: "
          f"{n_int} re-rendered as mentor-template twins, {n_fin} already answer-template twins)")
    assert n_int + n_fin == n_decoy, "decoy accounting mismatch"
    assert {r["passage_id"] for r in rows} == {r["passage_id"] for r in corpus}, "slug set changed!"


if __name__ == "__main__":
    main()
