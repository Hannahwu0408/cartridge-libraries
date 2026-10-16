"""analyze_chain_failures.py — stage-by-stage failure taxonomy for the iterative CHAIN controller.

The chain controller runs four stages; this attributes every failure to the stage that broke it:
  hop-1 retrieve (= dense@1)  →  bridge_extract  →  hop-2 retrieve (by bridge)  →  reframe + answer

Router records already store the whole trace (`bridge`, `answer_q`, `retrieved`, `pred`), so this is a
read-only analysis — no GPU, no re-run.

Failure classes:
  A BRIDGE_DEGENERATE  bridge empty / echoed the question / refusal text  → extraction stage broke
  B BRIDGE_WRONG       clean entity, but it matches no gold article       → extracted the wrong entity
  C RETRIEVAL_WRONG    bridge matches a gold, but hop-2 pulled a non-gold → namesake/ambiguity (disambiguation)
  D ANSWER_REFUSAL     right article, model says "not mentioned"          → extraction blocked
  E ANSWER_WRONG       right article, confident but wrong answer          → hallucination / wrong span
  F HOP1_WRONG         hop-1 never retrieved a gold article               → upstream retrieval miss

Usage:
  python scripts/analyze_chain_failures.py results/chain_ablation/iso/router_results.json \
      --articles-file data/2wiki_v2/articles.jsonl [--show 40] [--threshold 0.5]
"""
from __future__ import annotations

import re
import json
import argparse
from pathlib import Path

REFUSAL = ("not mentioned", "not explicitly", "no information", "does not mention", "cannot",
           "not stated", "not provided", "not specified", "unable to", "isn't mentioned",
           "is not available", "not found", "does not provide", "no mention", "not contain")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()


def toks(s: str) -> set[str]:
    """Content tokens (drop short/stop-ish words) for fuzzy title↔entity matching."""
    stop = {"the", "of", "and", "a", "an", "in", "on", "for", "de", "la", "von", "van", "jr", "sr",
            "st", "mr", "mrs", "dr", "sir", "lord", "lady", "king", "queen", "duke", "prince"}
    return {w for w in norm(s).split() if len(w) > 2 and w not in stop}


def matches(entity: str, title: str) -> bool:
    """Does the decoded bridge entity plausibly name this article?"""
    a, b = toks(entity), toks(title)
    if not a or not b:
        return False
    return len(a & b) >= min(2, min(len(a), len(b)))   # share ≥2 content tokens (or all, if shorter)


def cut(s, n=95):
    s = (s or "").replace("\n", " ").strip()
    return s[:n] + ("…" if len(s) > n else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--articles-file", default="data/2wiki_v2/articles.jsonl")
    ap.add_argument("--threshold", type=float, default=0.5, help="fuzzy below this = failure")
    ap.add_argument("--show", type=int, default=40, help="max per-question dumps")
    a = ap.parse_args()

    d = json.load(open(a.results))
    R = d["records"] if isinstance(d, dict) and "records" in d else d

    title = {}
    for line in open(a.articles_file, encoding="utf-8"):
        if line.strip():
            j = json.loads(line)
            title[j["passage_id"]] = j.get("title", j["passage_id"])

    def gold_of(r):
        g = r.get("gold") or r.get("article_slugs") or []
        if isinstance(g, dict):
            g = list(g.keys())
        return set(g)

    buckets: dict[str, list] = {k: [] for k in
                                ["A_BRIDGE_DEGENERATE", "B_BRIDGE_WRONG", "C_RETRIEVAL_WRONG",
                                 "D_ANSWER_REFUSAL", "E_ANSWER_WRONG", "F_HOP1_WRONG"]}
    n_total = n_fail = 0
    hop1_hit = hop2_hit = bridge_ok = n_hop1_known = 0

    for r in R:
        ch = r.get("policies", {}).get("chain")
        if not ch:
            continue
        n_total += 1
        gold = gold_of(r)
        # hop-1 is only recoverable from the dense@1 policy (chain doesn't store its own hop-1).
        # If the run lacked --modes dense, hop-1 is UNKNOWN — do not mislabel failures as hop-1 misses.
        d1 = r.get("policies", {}).get("dense@1")
        has_hop1 = d1 is not None
        hop1 = set(d1.get("retrieved", [])) if has_hop1 else set()
        hop2 = set(ch.get("retrieved", []))
        bridge = (ch.get("bridge") or "").strip()
        pred = ch.get("pred") or ""
        fuzzy = ch.get("fuzzy", 0.0)

        h1_ok = bool(hop1 & gold) if has_hop1 else True   # unknown → assume ok for attribution
        h2_ok = bool(hop2 & gold)
        b_ok = any(matches(bridge, title.get(s, s)) for s in gold) if bridge else False
        if has_hop1:
            hop1_hit += h1_ok
            n_hop1_known += 1
        hop2_hit += h2_ok
        bridge_ok += b_ok

        if fuzzy >= a.threshold:
            continue
        n_fail += 1

        rec = dict(id=r.get("id", ""), q=r.get("question", ""), gold_ans=r.get("answer", ""),
                   gold=[title.get(s, s) for s in gold], bridge=bridge,
                   hop1=[title.get(s, s) for s in hop1], hop2=[title.get(s, s) for s in hop2],
                   aq=ch.get("answer_q", ""), pred=pred, fuzzy=fuzzy,
                   teacher=r.get("teacher", {}).get("fuzzy"))

        low = pred.lower()
        nb = norm(bridge)
        degenerate = (not bridge) or len(nb.split()) > 8 or any(p in bridge.lower() for p in REFUSAL)

        # Attribute by OUTCOME first: if hop-2 landed on a gold article the bridge did its job
        # (dense retrieval is forgiving of spelling/alias mismatch), so the failure is downstream.
        if not h1_ok:
            buckets["F_HOP1_WRONG"].append(rec)
        elif h2_ok:                                   # reached a gold article — failure is at answer time
            buckets["D_ANSWER_REFUSAL" if any(p in low for p in REFUSAL) else "E_ANSWER_WRONG"].append(rec)
        elif degenerate:                              # never reached gold, and the bridge was malformed
            buckets["A_BRIDGE_DEGENERATE"].append(rec)
        elif b_ok:                                    # bridge named a gold, but retrieval went elsewhere
            buckets["C_RETRIEVAL_WRONG"].append(rec)
        else:                                         # bridge named a non-gold entity → wrong entity decoded
            buckets["B_BRIDGE_WRONG"].append(rec)

    print(f"\n===== CHAIN FAILURE TAXONOMY — {a.results} =====")
    print(f"{n_total} questions · {n_total - n_fail} pass / {n_fail} fail (fuzzy < {a.threshold})")
    h1_str = (f"hop-1 retrieved a gold {hop1_hit}/{n_hop1_known} ({hop1_hit/max(n_hop1_known,1):.0%})"
              if n_hop1_known else "hop-1 UNKNOWN (run router with --modes dense,chain to attribute it)")
    print(f"stage hit-rates: {h1_str} · "
          f"bridge names a gold {bridge_ok}/{n_total} ({bridge_ok/max(n_total,1):.0%}) · "
          f"hop-2 retrieved a gold {hop2_hit}/{n_total} ({hop2_hit/max(n_total,1):.0%})")
    print("\n--- failure breakdown ---")
    for k, v in buckets.items():
        share = len(v) / max(n_fail, 1)
        print(f"  {k:22s} {len(v):3d}  ({share:5.1%} of failures)")

    # teacher-solvable share: failures the text teacher DOES get = our real headroom
    solvable = sum(1 for v in buckets.values() for r in v if (r["teacher"] or 0) >= a.threshold)
    print(f"\n  of {n_fail} failures, teacher solves {solvable} → recoverable headroom "
          f"({solvable/max(n_fail,1):.0%}); the rest are dataset/metric-hard")

    print("\n" + "=" * 100)
    shown = 0
    for k, v in buckets.items():
        if not v:
            continue
        print(f"\n########## {k}  (n={len(v)}) ##########")
        for r in v:
            if shown >= a.show:
                break
            shown += 1
            print(f"\n[{r['id'][:10]}] {cut(r['q'], 105)}")
            print(f"   GOLD ans : {r['gold_ans']}   | gold articles: {r['gold']}")
            print(f"   hop-1    : {r['hop1']}")
            print(f"   bridge   : {cut(r['bridge'], 70)!r}")
            print(f"   hop-2    : {r['hop2']}")
            print(f"   reframed : {cut(r['aq'], 90)}")
            print(f"   PRED     : {cut(r['pred'], 95)}  (fuzzy {r['fuzzy']:.2f}, teacher {r['teacher']})")


if __name__ == "__main__":
    main()
