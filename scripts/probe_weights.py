"""probe_weights.py — per-cartridge gnorm weight AT THE ANSWER POSITION, single-suffices vs genuine-2-hop.

Directly tests the two dilution hypotheses on a trained 2Wiki pool:
  (1) is the ANSWER-bearing cartridge down-weighted under composition? (weight fraction < 0.5)
  (2) is the QUERY-NAMED cartridge up-weighted? (inverted relevance — the query surface-matches the wrong
      cartridge, per S1c: "why did Kirill's wife die" matches Kirill, answer in the wife's article)
Reuses group_weights() from evaluate_composition (weights at the generation position: gen_mean + gen_topq,
the top-quartile that surfaces the heads doing the hop). Buckets come from a router_results.json.

Usage:
  python scripts/probe_weights.py --router-results results/2wiki_k/base1k/router_results.json \
      --multihop-file data/2wiki_v2/multihop.jsonl --articles-file data/2wiki_v2/articles.jsonl \
      --cartridge-dir cartridges/2wiki_v3_base1k --cartridge-len 128
"""
from __future__ import annotations

import re
import sys
import json
import argparse
import statistics
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cartridge_qwen import load_model, QwenCartridge, get_device   # noqa: E402
from evaluate_composition import group_weights                     # noqa: E402


def recs(p):
    d = json.load(open(p))
    return d["records"] if isinstance(d, dict) and "records" in d else d


_HONORIFIC = {"grand", "duke", "duchess", "prince", "princess", "lord", "lady", "sir", "king",
              "queen", "count", "countess", "baron", "baroness", "earl", "marquis", "marquess",
              "emperor", "empress", "saint", "st", "archbishop", "pope", "dame", "the", "of",
              "infante", "seigneur", "margrave"}


def title_head(title: str) -> str:
    """distinctive name token — the first non-honorific word (e.g. 'Grand Duke Kirill…' -> 'kirill')."""
    t = re.sub(r"\(.*?\)|,.*$", "", title)              # drop parenthetical + everything after first comma
    for w in re.findall(r"[A-Za-z][\w'-]{2,}", t):
        if w.lower() not in _HONORIFIC:
            return w.lower()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router-results", required=True)
    ap.add_argument("--multihop-file", required=True)
    ap.add_argument("--articles-file", required=True)
    ap.add_argument("--cartridge-dir", required=True)
    ap.add_argument("--cartridge-len", type=int, default=128)
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-per-bucket", type=int, default=25)
    a = ap.parse_args()

    device = get_device()
    model, tok = load_model(a.model_id, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    mh = {q["id"]: q for q in (json.loads(l) for l in open(a.multihop_file) if l.strip())}
    title = {json.loads(l)["passage_id"]: json.loads(l).get("title", "")
             for l in open(a.articles_file) if l.strip()}
    R = recs(a.router_results)
    cd = Path(a.cartridge_dir)
    cache: dict = {}

    def load(slug):
        if slug not in cache:
            p = cd / slug / f"cartridge_{a.cartridge_len}.pt"
            cache[slug] = QwenCartridge.load(p, device=device) if p.exists() else None
        return cache[slug]

    def pol(r, l):
        return r["policies"].get(l, {})

    full = [r for r in R if pol(r, "dense@2").get("recall") == 1.0 and "single_best" in pol(r, "dense@2")]
    ss = [r for r in full if pol(r, "dense@2")["single_best"] >= 0.5]
    g2 = [r for r in full if pol(r, "dense@2")["single_best"] < 0.5 and r.get("teacher", {}).get("fuzzy", 0) >= 0.5]

    def probe(bucket, name):
        ans_fr, named_fr, examples = [], [], []
        for r in bucket[:a.max_per_bucket]:
            q = mh.get(r["id"])
            if not q:
                continue
            slugs = q.get("article_slugs", [])[:2]
            if len(slugs) != 2:
                continue
            carts = [load(s) for s in slugs]
            if any(c is None for c in carts):
                continue
            gw = group_weights(model, tok, q["question"], carts, device)
            if gw is None:
                continue
            sing = {s.get("slug"): s.get("fuzzy", 0) for s in pol(r, "dense@2").get("singles", [])}
            ans_idx = 0 if sing.get(slugs[0], 0) >= sing.get(slugs[1], 0) else 1
            qlow = q["question"].lower()
            named = [bool(title_head(title.get(s, "")) and title_head(title.get(s, "")) in qlow) for s in slugs]
            w = gw["gen_topq"]                                  # top-quartile = the decisive heads
            tot = w[0] + w[1]
            if tot <= 0:
                continue
            fa = w[ans_idx] / tot
            ans_fr.append(fa)
            if named[0] != named[1]:                            # exactly one cart is query-named
                ni = 0 if named[0] else 1
                named_fr.append(w[ni] / tot)
            examples.append((fa, w[ans_idx], w[1 - ans_idx], named[ans_idx], q["question"][:58]))
        print(f"  {name} (n={len(ans_fr)}):")
        if ans_fr:
            print(f"    ANSWER-cart weight fraction (gen_topq): {statistics.mean(ans_fr):.3f}   [<0.5 = answer cart DOWN-weighted]")
        if named_fr:
            print(f"    QUERY-NAMED-cart weight fraction      : {statistics.mean(named_fr):.3f}   [>0.5 = query-named up-weighted = inverted-relevance]")
        return examples

    print(f"\n=== WEIGHT PROBE  {a.cartridge_dir}  (single-suffices n={len(ss)}, genuine-2hop n={len(g2)}) ===")
    print("SINGLE-SUFFICES:")
    ex_ss = probe(ss, "single-suffices")
    print("GENUINE-2HOP:")
    ex_g2 = probe(g2, "genuine-2hop")

    print("\n--- per-question (answer-cart fraction; named=is answer cart the query-named one?) ---")
    for tag, ex in [("ss", ex_ss), ("g2", ex_g2)]:
        for fa, wa, wo, is_named, qt in sorted(ex)[:8]:
            print(f"  [{tag}] frac={fa:.2f} (ans={wa:.2f} oth={wo:.2f}) named={is_named}  {qt}")


if __name__ == "__main__":
    main()
