"""
classify_compose_failures.py — 12-I.2 failure-mode split (read-only over a saved eval).

The aggregate synth-rate hides WHY compose fails on a multi-subject-A cell. Two mechanisms, different fixes:
  • WRONG-INTERMEDIATE  — compose returns a DECOY master's town (one of the two non-answer towns in the
    roster). The bridge resolved to the WRONG entity and chained to its fact → "chaining-to-the-wrong-thing".
    Fix points at disambiguation / route-to-single.
  • FAILURE-TO-CHAIN     — compose returns no town at all (empty / refusal / an A-fact / a bare master name).
    It never resolved an intermediate → "failing-to-chain". Fix points at in-cartridge retrieval / the gate.

Reads the per-question `preds["compose_gnorm"]` already saved by evaluate_composition.py, and the roster's
three towns (gold + 2 decoys) via synth_rosters.get_rosters. No GPU, no re-run.

Usage:
  python scripts/classify_compose_failures.py \
      --results-file results/synth_i/amulti_fic/composition_results.json \
      --articles-file data/synth_amulti_fic/articles_mm.jsonl \
      --multihop-file data/synth_amulti_fic/multihop.jsonl
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_rosters import get_rosters, load_jsonl          # noqa: E402
from evaluate_squad import fuzzy_containment                # noqa: E402

HIT = 0.7   # fuzzy-containment threshold (matches the eval's fuzzy metric, tolerant of OOV spelling)


def main():
    ap = argparse.ArgumentParser(description="split compose failures: wrong-intermediate vs failure-to-chain")
    ap.add_argument("--results-file", required=True)
    ap.add_argument("--articles-file", required=True)
    ap.add_argument("--multihop-file", required=True)
    ap.add_argument("--condition", default="compose_gnorm")
    args = ap.parse_args()

    rosters = get_rosters(args.articles_file, args.multihop_file)
    mh = {q["id"]: q for q in load_jsonl(args.multihop_file)}
    records = json.loads(Path(args.results_file).read_text())["records"]

    n = correct = wrong_intermediate = failure_to_chain = other = 0
    examples = {"wrong_intermediate": [], "failure_to_chain": [], "other": []}
    for r in records:
        pred = r.get("preds", {}).get(args.condition)
        if pred is None or r["id"] not in mh:
            continue
        n += 1
        gold = r["answer"]
        e2_slug = mh[r["id"]]["article_slugs"][1]
        decoy_towns = [m["town"] for m in rosters[e2_slug] if m["town"] != gold]

        if fuzzy_containment(pred, gold) >= HIT:
            correct += 1
            continue
        hit_decoy = next((t for t in decoy_towns if fuzzy_containment(pred, t) >= HIT), None)
        if hit_decoy:
            wrong_intermediate += 1
            examples["wrong_intermediate"].append((r["id"], hit_decoy, pred[:80]))
        elif any(fuzzy_containment(pred, t) >= HIT for t in [gold] + decoy_towns):
            other += 1                                       # (shouldn't hit; gold handled above)
            examples["other"].append((r["id"], "", pred[:80]))
        else:
            failure_to_chain += 1                            # no town at all → never chained
            examples["failure_to_chain"].append((r["id"], "", pred[:80]))

    fails = wrong_intermediate + failure_to_chain + other
    print(f"\n{'='*66}\nCOMPOSE FAILURE MODES  ({Path(args.results_file).parent.name}, cond={args.condition}, n={n})")
    print(f"  correct                : {correct}/{n} = {correct/n:.2f}" if n else "  (no records)")
    print(f"  FAILURES               : {fails}/{n}")
    if fails:
        print(f"    wrong-intermediate   : {wrong_intermediate}/{fails} = {wrong_intermediate/fails:.2f}  "
              f"(returned a DECOY master's town → resolved to the wrong entity)")
        print(f"    failure-to-chain     : {failure_to_chain}/{fails} = {failure_to_chain/fails:.2f}  "
              f"(no town → never resolved an intermediate)")
        if other:
            print(f"    other                : {other}/{fails}")
    for k in ("wrong_intermediate", "failure_to_chain"):
        for rid, extra, p in examples[k][:4]:
            print(f"    [{k}] {rid} {('→'+extra) if extra else ''}  «{p}»")


if __name__ == "__main__":
    main()
