#!/usr/bin/env python3
"""oracle_patterns.py — where do our chain and a frontier oracle disagree, and why?

Free, local, zero API. Joins the oracle arms with the chain's per-question trace and asks:
is there structure in the questions Opus+gold solves and we miss (the recoverable headroom),
and in the ones we solve and Opus misses?

Correctness labels come from the BLIND AUDIT (judge_all) when available -- not raw fuzzy --
so the cells survive or fall on audited evidence. Falls back to raw fuzzy with a loud warning.

Hypotheses it is built to falsify (stated up front so a null is readable):
  H1  our "we beat the oracle" wins are verbose-containment false positives
  H2  our losses are dominated by cartridge REFUSALS => a single-cartridge fidelity problem,
      not a chain-control problem (would point the next experiment at self-study budget)
  H3  our wins concentrate in kinship questions, where R1's deterministic algebra fires
  H4  Opus's failures concentrate on idiosyncratic golds

Usage:  python scripts/oracle_patterns.py [--show-questions]
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

REFUSAL = ("does not mention", "not mentioned", "does not provide", "no mention",
           "not provided", "does not specify", "unable to", "cannot determine")

KINSHIP = re.compile(r"\b(father|mother|wife|husband|spouse|son|daughter|child|children|"
                     r"grand(?:father|mother|son|daughter)|step(?:father|mother)|"
                     r"sibling|brother|sister|paternal|maternal|married)\b", re.I)
WORK = re.compile(r"\b(director|directed|performer|performed|singer|award|film|movie|song|"
                  r"album|series|composer|producer|writer|author|screenwriter)\b", re.I)
PLACEDATE = re.compile(r"\b(born|birth|died|death|place of birth|place of death|when|where)\b", re.I)


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def relation_family(q):
    if KINSHIP.search(q):
        return "kinship"
    if WORK.search(q):
        return "creative-work"
    if PLACEDATE.search(q):
        return "place/date"
    return "other"


def gold_class(a):
    a = a.strip()
    if re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", a) or re.match(r"^\d", a):
        return "date/number"
    if re.search(r"\b(university|college|school|company|band|party|club|army|church|"
                 r"institute|museum|kingdom|dynasty)\b", a, re.I):
        return "organisation"
    if len(a.split()) <= 3 and a[:1].isupper():
        return "person/place"
    return "other"


def bar(label, cell_counts, cells, width=34):
    row = "  ".join(f"{cells[c]:>3}" for c in cell_counts)
    return f"  {label:<{width}} {row}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/claude_oracle")
    ap.add_argument("--router-results", default="results/chain_fix/R3/router_results.json")
    ap.add_argument("--multihop", default="data/2wiki_v2/multihop.jsonl")
    ap.add_argument("--show-questions", action="store_true",
                    help="print every question in the two disagreement cells")
    a = ap.parse_args()

    d = Path(a.results_dir)
    ceiling = {r["id"]: r for r in json.loads((d / "ceiling.json").read_text())["records"]}
    para = {r["id"]: r for r in json.loads((d / "parametric.json").read_text())["records"]}
    mh = {q["id"]: q for q in load_jsonl(a.multihop)}
    rr = {r["id"]: r for r in json.loads(Path(a.router_results).read_text())["records"]}

    # ── correctness labels: audited if we have them ──────────────────────────
    ja_path = d / "judge_all.json"
    audited = ja_path.exists()
    if audited:
        ja = json.loads(ja_path.read_text())
        ok = {(r["system"], r["id"]): r["verdict"] == "equivalent" for r in ja["records"]}
        verdict = {(r["system"], r["id"]): r["verdict"] for r in ja["records"]}
        C = {i: ok.get(("ceiling", i), False) for i in rr}
        K = {i: ok.get(("chain", i), False) for i in rr}
        broken = set(ja["gold_problem_ids"])
        fps = {r["id"] for r in ja["records"] if r["system"] == "chain" and r["false_positive"]}
        print(f"labels: BLIND AUDIT (judge_all) · {len(broken)} broken golds · "
              f"{len(fps)} chain false-positives caught")
    else:
        print("!! judge_all.json not found — falling back to RAW FUZZY labels.")
        print("!! H1 (our wins are verbose false positives) CANNOT be tested this way.")
        C = {i: ceiling[i]["fuzzy"] == 1.0 for i in rr}
        K = {i: rr[i]["policies"]["chain"]["fuzzy"] == 1.0 for i in rr}
        broken, fps, verdict = set(), set(), {}

    CELLS = ["both", "oracle-only", "ours-only", "neither"]

    def cell_of(i):
        return ("both" if C[i] and K[i] else "oracle-only" if C[i] else
                "ours-only" if K[i] else "neither")

    cell = {i: cell_of(i) for i in rr}
    counts = Counter(cell.values())
    n = len(rr)
    print(f"\n=== the 2x2, n={n} ({'audited' if audited else 'raw fuzzy'}) ===")
    for c in CELLS:
        print(f"  {c:<14} {counts[c]:>3}")
    assert sum(counts.values()) == n, "cells must reconcile to n"

    def breakdown(title, keyfn):
        groups = {}
        for i in rr:
            groups.setdefault(keyfn(i), Counter())[cell[i]] += 1
        print(f"\n--- {title} ---")
        print(f"  {'':<34} {'  '.join(f'{c[:9]:>3}' for c in CELLS)}")
        for g in sorted(groups, key=lambda g: -sum(groups[g].values())):
            row = groups[g]
            tot = sum(row.values())
            print(f"  {str(g)[:32]:<34} " +
                  "  ".join(f"{row[c]:>3}" for c in CELLS) + f"   (n={tot})")

    # ── features ─────────────────────────────────────────────────────────────
    def our_pred(i):
        return rr[i]["policies"]["chain"]["pred"] or ""

    def refused(i):
        p = our_pred(i).lower()
        return any(s in p for s in REFUSAL)

    def bridge_ok(i):
        """did the decoded bridge entity land on a gold article title?"""
        b = (rr[i]["policies"]["chain"].get("bridge") or "").lower()
        if not b.strip():
            return "no-bridge"
        toks = set(re.findall(r"[a-z0-9]+", b))
        golds = set(rr[i].get("gold", []))
        for g in golds:
            gt = set(re.findall(r"[a-z0-9]+", g.replace("_", " ")))
            if gt and (gt <= toks or toks <= gt or len(gt & toks) >= 2):
                return "bridge→gold"
        return "bridge≠gold"

    breakdown("question type", lambda i: mh[i]["type"])
    breakdown("relation family", lambda i: relation_family(mh[i]["question"]))
    breakdown("gold answer class", lambda i: gold_class(mh[i]["answer"]))
    breakdown("our retrieval recall", lambda i:
              f"recall={rr[i]['policies']['chain']['recall']}")
    breakdown("H2: our answer is a REFUSAL", lambda i: "refusal" if refused(i) else "asserted")
    breakdown("decoded bridge entity", bridge_ok)
    breakdown("route-to-single already solved it (dense@1)", lambda i:
              "single-suffices" if rr[i]["policies"].get("dense@1", {}).get("fuzzy") == 1.0
              else "needs the chain")
    breakdown("parametrically answerable (Opus no-context)", lambda i:
              "parametric ✓" if para[i]["fuzzy"] == 1.0 else "parametric ✗")
    breakdown("4B teacher solved it", lambda i:
              "teacher ✓" if (rr[i].get("teacher", {}).get("fuzzy") == 1.0) else "teacher ✗")

    # ── hypothesis read-outs ─────────────────────────────────────────────────
    ours_only = [i for i in rr if cell[i] == "ours-only"]
    oracle_only = [i for i in rr if cell[i] == "oracle-only"]
    print("\n=== hypothesis read-outs ===")
    if audited:
        print(f"  H1  chain false-positives caught by the audit: {len(fps)} "
              f"(of which in 'ours-only': {len(fps & set(ours_only))})")
        print(f"      => 'we beat the oracle' cell is {len(ours_only)} AFTER auditing.")
    print(f"  H2  refusals: ours-only {sum(refused(i) for i in ours_only)}/{len(ours_only)} · "
          f"oracle-only {sum(refused(i) for i in oracle_only)}/{len(oracle_only)}")
    kin = lambda ids: sum(relation_family(mh[i]['question']) == 'kinship' for i in ids)
    print(f"  H3  kinship share: ours-only {kin(ours_only)}/{len(ours_only)} · "
          f"oracle-only {kin(oracle_only)}/{len(oracle_only)} · "
          f"overall {kin(list(rr))}/{n}")
    empty_opus = sum(1 for i in oracle_only if not ceiling[i]["pred"].strip())
    print(f"  H4  broken golds: {len(broken)} · Opus emitted an EMPTY answer on "
          f"{sum(1 for i in rr if not ceiling[i]['pred'].strip())} questions")

    if a.show_questions:
        for name, ids in (("ORACLE-ONLY (recoverable headroom)", oracle_only),
                          ("OURS-ONLY (we beat the oracle)", ours_only),
                          ("NEITHER (unanswerable floor)",
                           [i for i in rr if cell[i] == "neither"])):
            print(f"\n=== {name} — {len(ids)} questions ===")
            for i in ids:
                print(f"\n  Q:     {mh[i]['question'][:100]}")
                print(f"  gold:  {mh[i]['answer']!r}   [{mh[i]['type']}, "
                      f"{relation_family(mh[i]['question'])}]")
                print(f"  ours:  {our_pred(i)[:100]!r}")
                print(f"  opus:  {ceiling[i]['pred'][:100]!r}")
                if verdict:
                    print(f"  verdicts: chain={verdict.get(('chain', i))} "
                          f"ceiling={verdict.get(('ceiling', i))}")


if __name__ == "__main__":
    main()
