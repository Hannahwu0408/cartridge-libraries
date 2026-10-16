"""continual_add.py — Phase 1 of the continual-add experiment: does GROWING a cartridge library degrade it?

Framing (why this is the whole test): the deployed system is route-to-single / chain over a FROZEN model with
each document an ISOLATED KV cartridge. So weight-level forgetting is impossible by construction — adding a
cartridge cannot change any other cartridge or the model. The ONLY coupling is the shared retrieval index, so the
sole way an add can hurt old questions is RETRIEVAL INTERFERENCE (a newcomer out-ranking an old gold). This driver
measures exactly that: it reveals the existing iso pool's cartridges in batches and, at each library size N, runs
the R3 chain over a FIXED early-question cohort (whose golds are always present) + the questions that first become
answerable at N. No training, no re-embedding — a sub-library index is just the full index.json filtered to the
present slugs (embeddings are over article text, independent of cartridge training).

Two arrival orders: `random` (average case) and `adversarial` (each early cohort's kNN confusers, from
topics.json, front-loaded → worst-case retrieval crowding).

Reads per N (from each router run's router_results.json): EARLY-cohort recall@1/@2 + chain fuzzy (the old-pool
regression curve); NEW-arrival chain fuzzy (first-class-arrival check).

Usage (runs router.py as a subprocess at each N — one nohup run):
  python scripts/continual_add.py \
    --index cartridges/2wiki_v3_iso/index.json \
    --multihop-file data/2wiki_v2/multihop.jsonl --articles-file data/2wiki_v2/articles.jsonl \
    --topics-file cartridges/2wiki_v3_iso/topics.json \
    --order random --schedule 0.25,0.5,0.75,1.0 --n-early 25 \
    --cartridge-len 128 --out-dir results/continual/random
"""
from __future__ import annotations

import sys
import json
import random
import argparse
import subprocess
from pathlib import Path

import numpy as np


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def golds_of(q):
    g = q.get("article_slugs") or q.get("gold") or []
    return [s for s in g]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="FULL iso index.json")
    ap.add_argument("--multihop-file", required=True)
    ap.add_argument("--articles-file", required=True)
    ap.add_argument("--topics-file", default=None, help="topics.json (knn_neighbors) — needed for adversarial")
    ap.add_argument("--order", choices=["random", "adversarial"], default="random")
    ap.add_argument("--schedule", default="0.25,0.5,0.75,1.0",
                    help="library sizes as fractions of |all slugs| (the seed floor is added automatically)")
    ap.add_argument("--n-early", type=int, default=25, help="size of the fixed early-question cohort")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cartridge-len", type=int, default=128)
    ap.add_argument("--compose-mode", default="gnorm")
    ap.add_argument("--gen-max-tokens", type=int, default=48)
    ap.add_argument("--out-dir", default="results/continual/run")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    idx = json.loads(Path(a.index).read_text())
    entries = idx.get("cartridges") or idx.get("entries")
    all_slugs = [e.get("id", e.get("passage_id")) for e in entries]
    slug_set = set(all_slugs)
    emb = {e.get("id", e.get("passage_id")): np.asarray(e["embedding"], dtype=np.float32) for e in entries}
    questions = load_jsonl(a.multihop_file)
    # keep only fully-answerable questions (both golds have a cartridge)
    qs = [q for q in questions if len(golds_of(q)) >= 2 and all(s in slug_set for s in golds_of(q)[:2])]

    rng = random.Random(a.seed)

    # 1) FIX the early cohort BEFORE ordering (avoids circularity): sample n-early questions; their gold
    #    slugs form the seed library S0 (present at EVERY N so their retrieval faces the growing competition).
    early_qs = rng.sample(qs, min(a.n_early, len(qs)))
    early_ids = {q["id"] for q in early_qs}
    S0 = sorted({s for q in early_qs for s in golds_of(q)[:2]})
    remaining = [s for s in all_slugs if s not in set(S0)]

    # 2) arrival order over the remaining slugs
    if a.order == "adversarial":
        if not a.topics_file:
            ap.error("--order adversarial needs --topics-file")
        topics = json.loads(Path(a.topics_file).read_text())
        knn = topics.get("knn_neighbors", {})
        confusers = set()
        for s in S0:                                   # front-load the early cohort's nearest neighbours
            for nb, _sim in knn.get(s, []):
                if nb in set(remaining):
                    confusers.add(nb)
        # score by max cosine similarity to any S0 slug (closest confusers first), tie-broken randomly
        S0mat = np.stack([emb[s] for s in S0]) if S0 else np.zeros((1, next(iter(emb.values())).shape[0]))
        def adv_score(s):
            base = 1.0 if s in confusers else 0.0
            return base + float(np.max(S0mat @ emb[s]))
        rng.shuffle(remaining)
        remaining.sort(key=adv_score, reverse=True)
    else:
        rng.shuffle(remaining)

    order = S0 + remaining                             # S0 always first → early cohort always answerable
    n_seed = len(S0)
    total = len(order)
    fracs = [float(x) for x in a.schedule.split(",")]
    Ns = sorted({max(n_seed, min(total, round(f * total))) for f in fracs} | {total})

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[continual_add] order={a.order} seed_lib={n_seed} slugs, {len(early_qs)} early Qs, "
          f"total={total}, N schedule={Ns}")

    # 3) at each N: write the sub-index (filter to the first-N slugs) and run the R3 chain over ALL questions.
    per_n = {}
    arrival_N = {}                                     # question id → the smallest N at which both golds present
    for N in Ns:
        present = set(order[:N])
        for q in qs:
            if q["id"] not in arrival_N and all(s in present for s in golds_of(q)[:2]):
                arrival_N[q["id"]] = N
        sub = {"embedder": idx.get("embedder"), "query_instruction": idx.get("query_instruction"),
               "cartridges": [e for e in entries if e.get("id", e.get("passage_id")) in present]}
        sub_path = out_dir / f"subindex_N{N}.json"
        sub_path.write_text(json.dumps(sub))
        rdir = out_dir / f"N{N}"
        cmd = [a.python, "scripts/router.py", "--index", str(sub_path),
               "--multihop-file", a.multihop_file, "--articles-file", a.articles_file,
               "--cartridge-len", str(a.cartridge_len), "--compose-mode", a.compose_mode,
               "--gen-max-tokens", str(a.gen_max_tokens), "--modes", "dense,chain", "--top-k", "1,2",
               "--chain-template", "--chain-overhop-guard", "--results-dir", str(rdir)]
        print(f"\n[continual_add] N={N} ({len(present)} carts) → running router …")
        subprocess.run(cmd, check=True)
        per_n[N] = json.loads((rdir / "router_results.json").read_text())

    # 4) trajectory analysis
    def recs(d):
        return d["records"] if isinstance(d, dict) and "records" in d else d

    def pol(r, l):
        return r.get("policies", {}).get(l, {})

    print(f"\n{'='*78}\nCONTINUAL-ADD TRAJECTORY  (order={a.order})\n{'='*78}")
    print(f"{'N':>5} {'carts':>6} | EARLY cohort (fixed, golds always present)      | NEW arrivals")
    print(f"{'':>5} {'':>6} | {'recall@1':>9} {'recall@2':>9} {'chain':>7} {'n':>4} | {'chain':>7} {'n':>4}")
    traj = []
    for N in Ns:
        R = recs(per_n[N])
        early = [r for r in R if r.get("id") in early_ids]
        e_r1 = mean([pol(r, "dense@1").get("recall", 0) for r in early])
        e_r2 = mean([pol(r, "dense@2").get("recall", 0) for r in early])
        e_ch = mean([pol(r, "chain").get("fuzzy", 0) for r in early])
        new_ids = {qid for qid, an in arrival_N.items() if an == N}
        new = [r for r in R if r.get("id") in new_ids]
        n_ch = mean([pol(r, "chain").get("fuzzy", 0) for r in new])
        print(f"{N:>5} {len(order[:N]):>6} | {e_r1:>9.3f} {e_r2:>9.3f} {e_ch:>7.3f} {len(early):>4} | "
              f"{n_ch:>7.3f} {len(new):>4}")
        traj.append({"N": N, "n_carts": len(order[:N]), "early_recall1": e_r1, "early_recall2": e_r2,
                     "early_chain": e_ch, "new_chain": n_ch, "n_early": len(early), "n_new": len(new)})

    e0, eN = traj[0]["early_chain"], traj[-1]["early_chain"]
    print(f"\nold-pool regression (early chain): {e0:.3f} at N={Ns[0]} → {eN:.3f} at N={Ns[-1]}  "
          f"(Δ {eN - e0:+.3f})  [~0 ⇒ adds don't degrade the library]")
    (out_dir / "trajectory.json").write_text(json.dumps(
        {"order": a.order, "seed_lib": n_seed, "early_ids": sorted(early_ids), "trajectory": traj}, indent=2))
    print(f"wrote {out_dir}/trajectory.json")


if __name__ == "__main__":
    main()
