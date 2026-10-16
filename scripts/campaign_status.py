"""campaign_status.py — what is done, what is free to claim, and how long the rest will take.

The scaled campaign runs across several machines with STATICALLY DISJOINT shard lists (no coordinator,
so no possibility of two workers doing the same article). This is how you see the board before
assigning the next worker, and how you check progress in the morning.

Reads, for each shard: the saturation ledger (results/campaign/saturation_sNofM.json) for generation,
and the cartridge directory for training. Both are ground truth written by the work itself — nothing
here depends on a log being parsed correctly.

Usage:
  python scripts/campaign_status.py
  python scripts/campaign_status.py --shards 60 --free 5      # suggest the next 5 free shards
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# measured rates, 4090 / 5090, at the settled 16:4 settings
GEN_4090 = 2326 * 1.95
TRAIN_4090_SPS = 9.3
PP, EPOCHS = 2000, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/2wiki_scale")
    ap.add_argument("--out-dir", default="results/campaign")
    ap.add_argument("--cart-dir", default="cartridges/2wiki_scale")
    ap.add_argument("--shards", type=int, default=60)
    ap.add_argument("--free", type=int, default=0, help="suggest this many unclaimed shards")
    a = ap.parse_args()

    arts = [json.loads(l)["passage_id"]
            for l in open(Path(a.data_dir) / "articles.jsonl", encoding="utf-8") if l.strip()]
    n_all, N = len(arts), a.shards
    cart = Path(a.cart_dir)
    trained = {p.parent.name for p in cart.rglob("cartridge_128.pt")} if cart.exists() else set()

    print(f"pool: {n_all} gold articles · {N} shards · {n_all/N:.1f} articles/shard\n")
    print(f"{'shard':>8}{'articles':>10}{'generated':>12}{'trained':>10}  state")
    gen_done = trn_done = 0
    free = []
    for i in range(1, N + 1):
        slugs = [s for k, s in enumerate(arts, start=1) if k % N == i % N]
        sat = Path(a.out_dir) / f"saturation_s{i}of{N}.json"
        made = 0
        if sat.exists():
            try:
                rows = json.load(open(sat))
                made = sum(1 for r in rows if r["stalled"] or r["made"] >= r["target"])
            except json.JSONDecodeError:
                made = -1                      # corrupt ledger: delete it, --resume redoes those
        t = sum(1 for s in slugs if s in trained)
        gen_done += max(made, 0)
        trn_done += t
        if made >= len(slugs) and t >= len(slugs):
            state = "COMPLETE"
        elif made >= len(slugs):
            state = "generated, needs training"
        elif made > 0:
            state = "generating (partial)"
        elif made < 0:
            state = "LEDGER CORRUPT — delete it"
        else:
            state = "free"
            free.append(f"{i}/{N}")
        print(f"{i:>6}/{N:<2}{len(slugs):>10}{max(made,0):>12}{t:>10}  {state}")

    print(f"\ngenerated {gen_done}/{n_all} articles · trained {trn_done}/{n_all} cartridges"
          f" · {len(free)} shards free")
    rem_gen, rem_trn = n_all - gen_done, n_all - trn_done
    gh = rem_gen * PP / GEN_4090
    th = rem_trn * PP * EPOCHS / (TRAIN_4090_SPS * 3600)
    print(f"remaining on ONE 4090: gen {gh/24:.1f} d + train {th/24:.1f} d = {(gh+th)/24:.1f} d")
    print("  (a second GPU roughly halves this; training never benefits from generation speedups)")
    if a.free and free:
        print(f"\nnext {min(a.free, len(free))} free shards to claim:  {' '.join(free[:a.free])}")
        print("  assign DISJOINT lists to different workers — there is no coordinator, and static")
        print("  disjoint assignment is what makes double-work impossible.")


if __name__ == "__main__":
    main()
