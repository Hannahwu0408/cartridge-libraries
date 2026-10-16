"""
build_topic_layer.py — CL-0: persist the topic/community layer over a cartridge index (CPU-only).

Reads index.json (from build_cartridge_index.py: one entry per cartridge with a FROZEN-embedder
dense embedding of its source article), fits a GMM over the embeddings (PCA-reduced — 196 samples
at 1024-dim would degenerate a raw GMM), sweeps K by BIC, and writes a topics.json sidecar:

    { K, pca_dim, bic_by_k, centroids_pca, regions: {id -> {soft, hard, entropy}},
      region_members, join_thr (per-region 95th-pct member cosine-distance to centroid),
      thresholds: join a region if cos-dist <= join_thr[r]; NOVEL (spawn) if farther than
      join_thr of EVERY region }

The topic layer is over ARTICLE TEXT embeddings, so it transfers unchanged across cartridge pools
trained on the same articles (base150 / desc150 / v3). Used by: CL-1 (--distractor-mode local),
CL-2 (join/spawn decisions), entropy diagnostic (multi-topic section candidates).

Usage:
  python scripts/build_topic_layer.py --index cartridges/2wiki_v2_base150/index.json \
      --out cartridges/2wiki_v2_base150/topics.json --k-sweep 6,8,12,16
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def entropy_norm(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def main():
    ap = argparse.ArgumentParser(description="CL-0 topic/community layer (CPU)")
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", default=None, help="default: topics.json next to the index")
    ap.add_argument("--k-sweep", default="6,8,12,16")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--join-pct", type=float, default=95.0,
                    help="per-region join threshold = this percentile of member distances")
    ap.add_argument("--knn", type=int, default=10,
                    help="per-cartridge nearest neighbors to store (CL-1 hard-negative distractors); "
                         "kNN-adjacent carts = same-family confusables, so these ARE the hard negatives")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-members", type=int, default=8)
    a = ap.parse_args()

    idx = json.loads(Path(a.index).read_text())
    if isinstance(idx, dict):                       # build_cartridge_index writes {"cartridges": [...]}
        entries = idx.get("cartridges") or idx.get("entries")
    else:
        entries = idx                               # bare list fallback
    if not entries:
        raise SystemExit(f"no cartridge entries found in {a.index} (keys: "
                         f"{list(idx)[:6] if isinstance(idx, dict) else type(idx).__name__})")
    ids = [e["id"] if "id" in e else e["passage_id"] for e in entries]
    titles = {ids[i]: entries[i].get("title", ids[i]) for i in range(len(entries))}
    X = np.array([e["embedding"] for e in entries], dtype=np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    n, d = X.shape
    print(f"index: {n} cartridges, dim {d}")

    pca_dim = min(a.pca_dim, n - 1, d)
    P = PCA(n_components=pca_dim, random_state=a.seed).fit(X)
    Z = P.transform(X)
    print(f"PCA -> {pca_dim} dims (explained var {P.explained_variance_ratio_.sum():.2f})")

    ks = [int(k) for k in a.k_sweep.split(",") if 1 < int(k) < n]
    fits = {}
    for k in ks:
        g = GaussianMixture(n_components=k, covariance_type="diag", n_init=3,
                            random_state=a.seed).fit(Z)
        fits[k] = (g, g.bic(Z))
    bic_by_k = {k: round(b, 1) for k, (_, b) in fits.items()}
    K = min(bic_by_k, key=bic_by_k.get)
    gmm = fits[K][0]
    print(f"BIC by K: {bic_by_k}  ->  K = {K}")

    soft = gmm.predict_proba(Z)                                    # (n, K)
    hard = soft.argmax(axis=1)
    ent = np.array([entropy_norm(p) for p in soft])

    # centroids back in embedding space (mean of members' normalized embeddings) for cosine geometry
    cent = np.zeros((K, d))
    for r in range(K):
        m = X[hard == r]
        cent[r] = m.mean(axis=0) if len(m) else 0.0
        cent[r] /= np.linalg.norm(cent[r]) + 1e-12
    cosdist = 1.0 - X @ cent.T                                     # (n, K)

    join_thr = {}
    for r in range(K):
        dists = cosdist[hard == r, r]
        join_thr[r] = float(np.percentile(dists, a.join_pct)) if len(dists) else 1.0

    region_members = {r: [ids[i] for i in np.where(hard == r)[0]] for r in range(K)}

    # per-cartridge kNN neighbors on the FULL embedding (X normalized -> dot = cosine). These are the
    # CL-1 hard-negative distractors: kNN-adjacent = same-family confusable (see the K=16 report). Stored
    # with similarity so CL-1 can sample distance-weighted. K-independent (doesn't use the GMM regions).
    kk = min(a.knn, n - 1)
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)
    knn_neighbors = {}
    for i in range(n):
        order = np.argpartition(-sim[i], kk)[:kk]
        order = order[np.argsort(-sim[i][order])]
        knn_neighbors[ids[i]] = [[ids[j], round(float(sim[i][j]), 4)] for j in order]

    out = {
        "index": str(a.index), "n": n, "K": K, "pca_dim": pca_dim, "bic_by_k": bic_by_k,
        "join_pct": a.join_pct, "knn": kk,
        "centroids_emb": cent.tolist(),
        "join_thr": {str(r): join_thr[r] for r in range(K)},
        "novelty_rule": "cosine distance to EVERY centroid > its join_thr -> spawn new region",
        "regions": {ids[i]: {"soft": [round(float(x), 4) for x in soft[i]],
                             "hard": int(hard[i]), "entropy": round(float(ent[i]), 4)}
                    for i in range(n)},
        "region_members": {str(r): m for r, m in region_members.items()},
        "knn_neighbors": knn_neighbors,
    }
    out_path = Path(a.out) if a.out else Path(a.index).parent / "topics.json"
    out_path.write_text(json.dumps(out))
    print(f"wrote {out_path}")

    print(f"\n===== COMMUNITY REPORT (K={K}) =====")
    for r in range(K):
        mem = region_members[r]
        me = ent[hard == r].mean() if len(mem) else 0.0
        print(f"\nregion {r}: {len(mem)} carts  (mean entropy {me:.2f}, join_thr {join_thr[r]:.3f})")
        for s in mem[: a.show_members]:
            print(f"   - {titles[s]}")
        if len(mem) > a.show_members:
            print(f"   ... +{len(mem) - a.show_members} more")

    print("\n===== TOP-10 HIGHEST-ENTROPY (multi-topic / split candidates) =====")
    for i in np.argsort(-ent)[:10]:
        print(f"  {ent[i]:.3f}  {titles[ids[i]]}")
    print(f"\nmean entropy {ent.mean():.3f} | singleton regions: "
          f"{sum(1 for r in range(K) if len(region_members[r]) == 1)}")

    print(f"\n===== kNN NEIGHBORS (top {min(4, kk)}) — sanity: are neighbors same-family confusables? =====")
    for i in np.argsort(-ent)[:6]:                       # highest-entropy (boundary) carts are the interesting ones
        nb = ", ".join(f"{titles[j]} ({s:.2f})" for j, s in knn_neighbors[ids[i]][:4])
        print(f"  {titles[ids[i]]}\n     -> {nb}")


if __name__ == "__main__":
    main()
