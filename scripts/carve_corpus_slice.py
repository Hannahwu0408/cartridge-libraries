"""carve_corpus_slice.py — cut a shard's own articles out of the 2000/pp corpus.

WHY. The 4000/pp extension needs each machine to know which questions were ALREADY generated for
its articles (`generate_selfstudy_data.py --seen-from`), and later to merge old+new. But the merged
corpus is ~392k conversations / ~200 MB, and the link to the rented pods has been slow on this
project. A machine only ever touches its OWN articles, so ship it only those: a 3/15 shard is ~40
articles ~ 15 MB.

The slug selection is byte-identical to the drivers' (`awk 'NR % n == i % n'` over articles.jsonl in
file order), so the slice provably matches what the machine will generate and train.

Usage:
  # one slice
  python scripts/carve_corpus_slice.py --shards 4/15 5/15 6/15 7/15 \
      --out data/2wiki_v2/ss_pool196_2000pp.podA.jsonl
  # all four machines at once (writes .earth/.podA/.podB/.podC)
  python scripts/carve_corpus_slice.py --split-plan
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# machine -> shards, mirroring the plan's 15-way weighted split (4090 : 5090 = 1 : 1.33)
PLAN = {"earth": ["1/15", "2/15", "3/15"],
        "podA":  ["4/15", "5/15", "6/15", "7/15"],
        "podB":  ["8/15", "9/15", "10/15", "11/15"],
        "podC":  ["12/15", "13/15", "14/15", "15/15"]}


def slugs_for(shards, articles_file):
    """same selection the shell drivers use: awk 'NR % n == i % n' over articles.jsonl order"""
    all_slugs = [json.loads(l)["passage_id"]
                 for l in open(articles_file, encoding="utf-8") if l.strip()]
    keep = set()
    for spec in shards:
        i, n = (int(x) for x in spec.split("/"))
        keep |= {s for k, s in enumerate(all_slugs, start=1) if k % n == i % n}
    return all_slugs, keep


def carve(corpus, keep, out_path):
    per = collections.Counter()
    with open(out_path, "w", encoding="utf-8") as out:
        for line in open(corpus, encoding="utf-8"):
            if not line.strip():
                continue
            pid = json.loads(line)["passage_id"]
            if pid in keep:
                per[pid] += 1
                out.write(line)
    missing = keep - set(per)
    assert not missing, (f"{len(missing)} article(s) of this shard are absent from the corpus "
                         f"(e.g. {sorted(missing)[:3]}) — carving a slice that cannot seed --seen-from")
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/2wiki_v2/ss_pool196_2000pp.jsonl")
    ap.add_argument("--articles-file", default="data/2wiki_v2/articles.jsonl")
    ap.add_argument("--shards", nargs="+", default=None, metavar="i/n")
    ap.add_argument("--out", default=None)
    ap.add_argument("--split-plan", action="store_true",
                    help=f"carve every machine's slice at once: {PLAN}")
    a = ap.parse_args()
    if not a.split_plan and not (a.shards and a.out):
        ap.error("give --shards ... --out FILE, or --split-plan")

    jobs = ([(m, sh, str(Path(a.corpus).with_suffix("")) + f".{m}.jsonl")
             for m, sh in PLAN.items()]
            if a.split_plan else [("slice", a.shards, a.out)])

    all_slugs, covered = None, set()
    for name, shards, out_path in jobs:
        all_slugs, keep = slugs_for(shards, a.articles_file)
        covered |= keep
        per = carve(a.corpus, keep, out_path)
        vals = per.values()
        print(f"  {name:<6} shards {','.join(shards):<24} {len(per):>3} articles · "
              f"{sum(vals):>7} convs (min/max {min(vals)}/{max(vals)}) → {out_path}")

    if a.split_plan:
        assert covered == set(all_slugs), (
            f"the split does not partition the pool: {len(set(all_slugs) - covered)} article(s) "
            f"assigned to NO machine — they would never be extended or retrained")
        print(f"  OK: {len(covered)} articles, every one assigned to exactly one machine")


if __name__ == "__main__":
    main()
