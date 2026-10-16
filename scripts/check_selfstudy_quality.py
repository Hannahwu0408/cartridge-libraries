"""
check_selfstudy_quality.py — data-quality gate for self-study jsonl BEFORE spending GPU-days training (12-K).

Reports, per file: per-passage yield vs --target, duplicate-question rate, empty/refusal-answer rate,
answer grounding vs the article context (hallucination proxy), and — with --desc — the NAME-LEAK rate
(a relational/descriptor conv whose question names its own subject defeats the descriptor framing).
Prints N random samples for human eyeballing. Exit code 0 = all gates pass, 1 = any gate fails, so a
driver script can gate on it (same halt-on-failure pattern as the 12-J DATA OK gate).

Usage:
  python scripts/check_selfstudy_quality.py --file data/2wiki_v2/selfstudy_v2_desc.jsonl \
      --articles-file data/2wiki_v2/articles.jsonl --target 100 --desc --samples 10
Gates (defaults): refusal <5%, dup <10%, grounding >60%, name-leak <25% (desc only),
                  yield: ≥90% of passages reach ≥70% of --target.
"""
from __future__ import annotations

import re
import json
import random
import argparse
import collections
from pathlib import Path

REFUSAL = ["not mentioned", "does not mention", "doesn't mention", "i don't know", "i do not know",
           "cannot", "can't answer", "no information", "not specified", "not provided", "unanswerable",
           "not stated", "unclear from"]
GENERIC = {"film", "song", "album", "band", "the", "and", "his", "her", "who", "what", "born", "from"}


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def title_tokens(title: str) -> list[str]:
    """distinctive tokens of the subject's title (parentheticals dropped, generics/short dropped)."""
    t = re.sub(r"\(.*?\)", " ", title)
    return [w for w in re.findall(r"[A-Za-z][\w'-]{3,}", t) if w.lower() not in GENERIC]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--articles-file", required=True)
    ap.add_argument("--target", type=int, required=True, help="intended convs/passage")
    ap.add_argument("--desc", action="store_true", help="relational file: also check subject name-leak")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-refusal", type=float, default=0.05)
    ap.add_argument("--max-dup", type=float, default=0.10)
    ap.add_argument("--min-grounding", type=float, default=0.60)
    ap.add_argument("--max-name-leak", type=float, default=0.25)
    a = ap.parse_args()
    random.seed(a.seed)

    arts = {r["passage_id"]: r for r in load(a.articles_file)}
    convs = load(a.file)
    n = len(convs)
    print(f"\n===== QUALITY: {a.file}  ({n} convs, target {a.target}/passage) =====")

    # per-passage yield
    byp = collections.Counter(c["passage_id"] for c in convs)
    counts = [byp.get(p, 0) for p in arts]
    ok_yield = sum(1 for c in counts if c >= 0.7 * a.target)
    print(f"yield/passage: min {min(counts)}  mean {sum(counts)/len(counts):.0f}  max {max(counts)}"
          f"  | passages ≥70% of target: {ok_yield}/{len(counts)}")

    # duplicates (within passage, lowercased question)
    seen, dup = set(), 0
    for c in convs:
        k = (c["passage_id"], c.get("question", "").strip().lower())
        dup += k in seen
        seen.add(k)

    # refusals / empty
    def bad_ans(ans):
        s = (ans or "").strip().lower()
        return len(s) < 2 or any(r in s for r in REFUSAL)
    refusals = sum(bad_ans(c.get("answer")) for c in convs)

    # grounding: ≥half of the answer's content words (len≥4, or digits) appear in the article context
    grounded = 0
    for c in convs:
        ctx = arts.get(c["passage_id"], {}).get("context", "").lower()
        words = re.findall(r"[a-z]{4,}|\d{2,}", (c.get("answer") or "").lower())
        if not words:
            grounded += 1                                    # no content words to check (e.g. "yes")
        else:
            grounded += (sum(w in ctx for w in words) >= 0.5 * len(words))

    # name-leak (desc only): question contains a distinctive token of its OWN subject's title
    leak = 0
    if a.desc:
        for c in convs:
            toks = title_tokens(arts.get(c["passage_id"], {}).get("title", ""))
            q = c.get("question", "").lower()
            leak += any(re.search(rf"\b{re.escape(t.lower())}\b", q) for t in toks)

    rows = [("refusal/empty answers", refusals / n, a.max_refusal, "<"),
            ("duplicate questions", dup / n, a.max_dup, "<"),
            ("answer grounding", grounded / n, a.min_grounding, ">")]
    if a.desc:
        rows.append(("subject NAME-LEAK", leak / n, a.max_name_leak, "<"))

    fails = 0
    for name, val, thr, d in rows:
        ok = val < thr if d == "<" else val > thr
        fails += not ok
        print(f"  {name:24s}: {val:6.1%}   (gate {d}{thr:.0%})  {'PASS' if ok else '*** FAIL ***'}")
    yield_ok = ok_yield >= 0.9 * len(counts)
    fails += not yield_ok
    print(f"  {'per-passage yield':24s}: {ok_yield}/{len(counts)} ≥70% of target  "
          f"{'PASS' if yield_ok else '*** FAIL ***'}")

    print(f"\n--- {a.samples} random samples (eyeball these) ---")
    for c in random.sample(convs, min(a.samples, n)):
        t = arts.get(c["passage_id"], {}).get("title", c["passage_id"])
        print(f"\n[{t}]\n  Q: {c.get('question','')[:180]}\n  A: {(c.get('answer') or '')[:180]}")

    print(f"\n===== {'ALL GATES PASS' if fails == 0 else f'{fails} GATE(S) FAILED'} =====")
    raise SystemExit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
