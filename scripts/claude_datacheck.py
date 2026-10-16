#!/usr/bin/env python3
"""claude_datacheck.py — are the SELF-STUDY answers factually supported by their own passage?

THE GATE BEFORE SCALING (CAS's data-quality lesson). Self-study answers are the distillation
TARGETS: whatever they assert is what the cartridge learns. If the generator hallucinated a birth
year, training on 2000 convs/passage entrenches the hallucination twice as hard. We already have
evidence this happens — the re-distilled cartridges summarise Ioan S. Nenițescu as born 1857 / 1861
while the passage says 1854 — so the question is whether the wrong fact came from the TRAINING DATA
or from lossy KV compression. Those need different fixes:
  wrong fact IS in the data  -> fix generation (judge-filter contradicted answers) BEFORE scaling
  wrong fact NOT in the data -> compression loss -> scaling self-study is the right lever

Two passes:
  1. RANDOM SAMPLE (~8 convs/article): overall supported / unsupported / contradicted rates.
  2. TARGETED: every conversation whose answer contains a KNOWN wrong value (WRONG_FACTS below),
     judged individually — the direct test of "is the hallucination in the data?"

Judging is blind to which cartridge/arm produced anything; the judge sees passage + Q + A only.
Responses are cached (shared Oracle cache), so re-runs are free and resumable.

Usage:
  export ANTHROPIC_API_KEY=...
  python scripts/claude_datacheck.py --selfstudy data/2wiki_v2/ss_fid30_subset.jsonl
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from claude_oracle import Oracle, safe_ask, run_parallel, parse_verdict, FAILED  # noqa: E402

# Wrong values we OBSERVED cartridges assert, paired with the truth from the passage. If a
# self-study answer contains the wrong value, the error was taught, not compressed.
WRONG_FACTS = {
    "ioan_s_neni_escu":            [("1857", "1854"), ("1861", "1854")],
    "william_walton":              [("London", "Ischia")],
    "robert_e_howard":             [("Brownwood", "Cross Plains")],
    "karan_johar":                 [("Hiroo Johar", "Yash Johar")],
    "leah_labelle":                [("Roderick Butler", "Rasual Butler")],
}

DATACHECK_SYSTEM = (
    "You are auditing training data. Given a source passage and a question-answer pair written "
    "about it, decide whether the ANSWER is factually supported BY THAT PASSAGE.\n"
    "Rules:\n"
    "- Judge only against the passage. Outside knowledge does not make an answer supported.\n"
    "- Minor paraphrase, summarising, or reasonable inference from stated facts = supported.\n"
    "- An answer that states a specific fact (a date, place, name, number) the passage does NOT "
    "state, or states differently, is contradicted — this is the case we care about most.\n"
    "- An answer that declines, hedges, or is too vague to check = unsupported (not contradicted).\n"
    'Reply with STRICT JSON only: {"verdict": "...", "reason": "..."} where verdict is one of '
    '"supported", "unsupported", "contradicted". reason: one short sentence.'
)
VERDICTS = ("supported", "unsupported", "contradicted")


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def judge_prompt(passage, q, a, max_ctx=6000):
    return (f"PASSAGE:\n{passage[:max_ctx]}\n\n"
            f"QUESTION: {q}\n\nANSWER TO AUDIT: {a}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfstudy", required=True, help="self-study jsonl subset for the testbed")
    ap.add_argument("--articles-file", default="data/2wiki_v2/articles.jsonl")
    ap.add_argument("--slugs-file", default="data/2wiki_v2/fidelity30_slugs.txt")
    ap.add_argument("--per-article", type=int, default=8, help="random convs sampled per article")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--results-dir", default="results/datacheck")
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--max-contradicted", type=float, default=0.05, help="GATE threshold")
    ap.add_argument("--cache-tag", default="",
                    help="appended to the cache key. REQUIRED when auditing a different generation "
                         "arm (e.g. --cache-tag q8a4): the key hashes only the QUESTION, so two "
                         "arms' different answers to the same question text would otherwise "
                         "silently share one cached verdict. Default empty keeps existing entries.")
    ap.add_argument("--emit-sample", default=None, metavar="PATH",
                    help="SELECT ONLY, no API calls: write just the convs this audit would judge "
                         "(the seeded per-article sample + every targeted wrong-value hit) and "
                         "exit. Run this on the machine holding the full self-study file, then "
                         "transfer the ~250-line output instead of the ~30k-line original — the "
                         "selection logic is identical, so the audit is unchanged.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    slugs = [s.strip() for s in Path(a.slugs_file).read_text().splitlines() if s.strip()]
    arts = {x["passage_id"]: x for x in load_jsonl(a.articles_file)}
    convs = [c for c in load_jsonl(a.selfstudy) if c["passage_id"] in set(slugs)]
    by_art = defaultdict(list)
    for c in convs:
        by_art[c["passage_id"]].append(c)
    print(f"{len(convs)} self-study convs across {len(by_art)}/{len(slugs)} testbed articles "
          f"(mean {len(convs)/max(1,len(by_art)):.0f}/article)")
    missing = [s for s in slugs if s not in by_art]
    if missing:
        print(f"  ⚠️  no convs found for {len(missing)} slugs: {missing[:4]}")

    rng = random.Random(0)
    sample = []
    for s, cs in by_art.items():
        sample += rng.sample(cs, min(a.per_article, len(cs)))

    # targeted: convs whose ANSWER contains a known-wrong value
    targeted = []
    for s, pairs in WRONG_FACTS.items():
        for c in by_art.get(s, []):
            for bad, true in pairs:
                if bad.lower() in (c.get("answer") or "").lower():
                    targeted.append({**c, "_bad": bad, "_true": true})
    print(f"random sample: {len(sample)} convs · targeted (contain a known wrong value): "
          f"{len(targeted)} convs\n")

    if a.emit_sample:
        seen, rows = set(), []
        for c in sample + targeted:
            k = (c["passage_id"], c["question"])
            if k not in seen:
                seen.add(k)
                rows.append({k2: c[k2] for k2 in ("passage_id", "seed_type", "question", "answer")
                             if k2 in c})
        p = Path(a.emit_sample)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"wrote {p} — {len(rows)} convs ({p.stat().st_size/1024:.0f} KB), no API calls made.\n"
              f"Transfer this file and run the audit on it; --per-article {a.per_article} will "
              f"then select all of them.")
        return

    out_dir = Path(a.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle = Oracle(a.model, out_dir / "cache.jsonl", dry_run=a.dry_run, max_tokens=1024)

    def judge(item):
        s = item["passage_id"]
        # STABLE key: Python's str hash() is salted per process, so hash()-based keys miss the
        # cache on every run — that is why 306 calls were re-issued (and 258 then died to
        # rate-limiting) instead of being free replays.
        qh = hashlib.sha1(item["question"].encode("utf-8")).hexdigest()[:12]
        tag = f":{a.cache_tag}" if a.cache_tag else ""
        raw = safe_ask(oracle, f"datacheck:{s}:{qh}:{a.model}{tag}",
                       DATACHECK_SYSTEM,
                       judge_prompt(arts[s]["context"], item["question"], item["answer"]))
        v, reason = parse_verdict(raw, allowed=VERDICTS)
        return {**{k: item[k] for k in ("passage_id", "seed_type", "question", "answer") if k in item},
                "verdict": v or "unparseable", "reason": reason,
                **({"_bad": item["_bad"], "_true": item["_true"]} if "_bad" in item else {})}

    print("── random-sample audit ──")
    recs = run_parallel(judge, sample, a.max_workers)
    counts = Counter(r["verdict"] for r in recs)
    n = len(recs)
    # Rate over PARSED verdicts only. Never let an unparseable/failed audit read as a low
    # contradiction rate — that is how the round-1 judge silently "passed" while 30 calls had
    # died. Coverage is reported and gates independently.
    parsed = n - counts["unparseable"]
    coverage = parsed / n if n else 0.0
    contra = counts["contradicted"] / parsed if parsed else 1.0
    for v in VERDICTS + ("unparseable",):
        print(f"  {v:<14} {counts[v]:>4}  ({counts[v]/max(1,n):.1%})")

    by_mode = defaultdict(Counter)
    for r in recs:
        by_mode[r.get("seed_type", "?")][r["verdict"]] += 1
    if len(by_mode) > 1:
        print("\n  by seed mode (contradicted / parsed):")
        for m, c in sorted(by_mode.items(), key=lambda kv: -kv[1]["contradicted"]):
            p = sum(c.values()) - c["unparseable"]
            print(f"    {m:<16} {c['contradicted']:>3}/{p:<4} ({c['contradicted']/max(1,p):.0%})")

    per_art = defaultdict(Counter)
    for r in recs:
        per_art[r["passage_id"]][r["verdict"]] += 1
    worst = sorted(per_art.items(), key=lambda kv: -kv[1]["contradicted"])[:6]
    print("\n  worst articles by contradiction count:")
    for s, c in worst:
        if c["contradicted"]:
            print(f"    {s[:44]:<44} {c['contradicted']}/{sum(c.values())}")

    print("\n  sample contradicted answers:")
    for r in [x for x in recs if x["verdict"] == "contradicted"][:6]:
        print(f"    [{r['passage_id'][:28]}] Q: {r['question'][:70]}")
        print(f"       A: {r['answer'][:90]}")
        print(f"       → {r['reason'][:130]}")

    trecs = []
    if targeted:
        print("\n── targeted audit: is the KNOWN wrong fact in the training data? ──")
        trecs = run_parallel(judge, targeted, a.max_workers)
        for r in trecs:
            print(f"  [{r['verdict']:>13}] {r['passage_id'][:26]:<26} wrong={r['_bad']!r} "
                  f"true={r['_true']!r}")
            print(f"       A: {r['answer'][:100]}")
    else:
        print("\n── targeted audit: NO self-study answer contains any known wrong value ──")
        print("   ⇒ the cartridges' wrong facts were NOT taught by this data; they are COMPRESSION")
        print("     loss, which is exactly what scaling self-study is supposed to fix.")

    MIN_COVERAGE = 0.90
    ok = coverage >= MIN_COVERAGE and contra <= a.max_contradicted
    print("\n" + "=" * 66)
    if coverage < MIN_COVERAGE:
        print(f"GATE  INDETERMINATE — only {coverage:.0%} of {n} audits returned a usable verdict "
              f"(need {MIN_COVERAGE:.0%}).\n      Re-run to retry the failures (responses are "
              f"cached); do NOT read the contradiction rate below as evidence.")
    print(f"GATE  contradicted {contra:.1%} of {parsed} parsed (threshold "
          f"{a.max_contradicted:.0%}, coverage {coverage:.0%}) → "
          f"{'PASS — proceed to the scaling arms' if ok else 'DO NOT SCALE YET'}")
    if targeted:
        n_bad = sum(1 for r in trecs if r["verdict"] == "contradicted")
        print(f"      {n_bad}/{len(trecs)} targeted convs confirmed as teaching a wrong fact "
              f"⇒ {'ADD a judge-filter to generation' if n_bad else 'wrong facts are not data-borne'}")
    if FAILED:
        print(f"      ⚠️  {len(FAILED)} calls failed (cached ones persist — re-run to retry)")

    suffix = f"_{a.cache_tag}" if a.cache_tag else ""
    out_json = out_dir / f"datacheck_results{suffix}.json"
    out_json.write_text(json.dumps(
        {"n_sampled": n, "counts": dict(counts), "contradicted_rate": contra, "coverage": coverage,
         "cache_tag": a.cache_tag, "selfstudy": a.selfstudy,
         "gate_pass": ok, "records": recs, "targeted": trecs}, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
