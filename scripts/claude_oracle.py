#!/usr/bin/env python3
"""claude_oracle.py — Claude-as-oracle upper-bound study (runs on the Mac, no GPU).

Three arms over the 100 2Wiki bridge questions:
  ceiling     Claude + BOTH gold articles in context  -> the true task ceiling
              (how much of the chain's 0.29 residual is dataset-hard vs 4B-reader weakness)
  parametric  Claude with NO context                  -> parametric/contamination floor
              (how much of 2Wiki a frontier model answers from weights alone)
  judge       Claude judges our 29 chain failures     -> metric-noise / genuine / gold_problem
              (externally verifies the "true accuracy ~0.74" claim)

Inputs (~1 MB total):
  data/2wiki_v2/multihop.jsonl            100 Qs: id, question, answer, article_slugs
  data/2wiki_v2/articles.jsonl            passage_id -> title, context
  results/chain_fix/R3/router_results.json  per-Q chain pred/fuzzy + teacher pred (the 0.71 run)

Guard: before any API call the script recomputes chain fuzzy from router_results and requires it
to match --expect-chain (0.71) — catches pulling the wrong run directory.

All responses are cached in <results-dir>/cache.jsonl keyed by (arm, question id, model), so
reruns cost nothing and a crashed run resumes mid-arm. Temperature 0 throughout.

Usage:
  export ANTHROPIC_API_KEY=...
  python scripts/claude_oracle.py --arm all
"""
import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_squad import containment, exact_match, f1_score, fuzzy_containment  # noqa: E402

CHAIN_LABEL = "chain"

SPAN_RULE = ("Reply with ONLY the shortest exact answer span — a name, date, place, or phrase. "
             "No explanation, no sentence, no punctuation beyond the span itself.")

CEILING_SYSTEM = ("You answer questions using ONLY the two articles provided. The answer is "
                  "always derivable from them (it may require combining a fact from each). "
                  + SPAN_RULE)

PARAMETRIC_SYSTEM = ("You answer questions from your own knowledge, with no reference material. "
                     + SPAN_RULE + " If you are not certain, give your single best guess — "
                     "never refuse and never say you don't know.")

JUDGE_SYSTEM = (
    "You are grading a question-answering system against a gold answer. Decide whether the "
    "system's prediction is correct in MEANING, ignoring surface form (case, articles, "
    "demonym vs country, minor spelling, extra words around the right span).\n"
    'Reply with STRICT JSON only: {"verdict": "...", "reason": "..."} where verdict is one of:\n'
    '  "equivalent"   - the prediction means the same as the gold answer (metric noise)\n'
    '  "wrong"        - the prediction is genuinely incorrect\n'
    '  "gold_problem" - the gold answer itself is wrong or ambiguous for the question as asked\n'
    "reason: one short sentence.")

# Round-2 blind audit protocol. Differences from JUDGE_SYSTEM, both deliberate:
#   - no hint about which system produced the prediction, and no reference answer (the round-1
#     prompt leaked the teacher's answer, and only for one of the two systems);
#   - explicit rule against the false-positive mode that `containment` cannot see — a verbose
#     prediction that MENTIONS the gold string while ASSERTING a different answer
#     ("Ashoka is from India. He ruled the Maurya empire." vs gold "Maurya dynasty").
JUDGE2_SYSTEM = (
    "You are grading one answer to a question against a gold answer. Judge the answer the text "
    "ACTUALLY ASSERTS, not whether the gold string happens to appear somewhere in it.\n"
    "Rules:\n"
    "- Ignore surface form: case, articles, punctuation, demonym vs country (French/France), "
    "abbreviation (US/United States), minor spelling, and extra words around a correct span.\n"
    "- If the text asserts one answer but merely mentions the gold in passing (background, a "
    "different entity, or a hedge), that is NOT correct.\n"
    "- If the text refuses or says the information is unavailable, that is NOT correct, even if it "
    "then speculates.\n"
    'Reply with STRICT JSON only: {"verdict": "...", "reason": "..."} where verdict is one of:\n'
    '  "equivalent"   - the asserted answer means the same as the gold\n'
    '  "wrong"        - the asserted answer is incorrect, or no answer is asserted\n'
    '  "gold_problem" - the gold itself is wrong or ambiguous for the question as asked\n'
    "reason: one short sentence.")

NHOP_SYSTEM = "Use the following passages to answer the question."


# ── data loading ─────────────────────────────────────────────────────────────

def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_inputs(args):
    questions = load_jsonl(args.multihop)
    articles = {a["passage_id"]: a for a in load_jsonl(args.articles)}
    rr = json.loads(Path(args.router_results).read_text())
    records = rr["records"]

    missing = [q["id"] for q in questions
               for s in q["article_slugs"] if s not in articles]
    if missing:
        sys.exit(f"ABORT: article slugs missing from {args.articles}: {sorted(set(missing))[:5]}")

    chain = {}
    for r in records:
        pol = r.get("policies", {}).get(CHAIN_LABEL)
        if pol is None:
            sys.exit(f"ABORT: record {r.get('id')} has no '{CHAIN_LABEL}' policy — wrong "
                     f"router_results.json? Policies present: {sorted(r.get('policies', {}))}")
        chain[r["id"]] = {"pred": pol["pred"], "fuzzy": pol["fuzzy"],
                          "teacher_pred": r.get("teacher", {}).get("pred"),
                          "teacher_fuzzy": r.get("teacher", {}).get("fuzzy")}

    chain_fuzzy = sum(c["fuzzy"] for c in chain.values()) / len(chain)
    if abs(chain_fuzzy - args.expect_chain) > 0.005:
        sys.exit(f"ABORT: recomputed chain fuzzy = {chain_fuzzy:.3f}, expected "
                 f"{args.expect_chain:.2f}. This looks like the wrong run — pull the "
                 f"router_results.json of the 0.71 R3 chain run (results/chain_fix/R3/), or "
                 f"pass --expect-chain to override.")
    teacher_vals = [c["teacher_fuzzy"] for c in chain.values() if c["teacher_fuzzy"] is not None]
    teacher_fuzzy = sum(teacher_vals) / len(teacher_vals) if teacher_vals else None
    print(f"loaded {len(questions)} questions · chain fuzzy {chain_fuzzy:.3f} ✓"
          + (f" · teacher fuzzy {teacher_fuzzy:.3f}" if teacher_fuzzy is not None else ""))
    return questions, articles, chain, chain_fuzzy, teacher_fuzzy


# ── Claude calls (cached, retried, temperature 0) ────────────────────────────

class Oracle:
    def __init__(self, model, cache_path, dry_run=False, max_tokens=200, max_attempts=8):
        self.model, self.max_tokens, self.dry_run = model, max_tokens, dry_run
        self.max_attempts = max_attempts
        self.cache_path = Path(cache_path)
        self.cache, self.lock = {}, threading.Lock()
        self.usage = {"input": 0, "output": 0, "calls": 0, "cache_hits": 0}
        # Newer models reject `temperature` outright; older ones need it pinned to 0 for
        # determinism. Send it, and drop it permanently the first time the API refuses it.
        self.send_temperature = True
        if self.cache_path.exists():
            for row in load_jsonl(self.cache_path):
                self.cache[row["key"]] = row["response"]
        self._client = None
        if not dry_run:
            import anthropic
            # BOUND EVERY ATTEMPT. Without an explicit timeout the SDK waits 10 MINUTES per request
            # and retries twice itself; wrapped in our own 8-attempt ladder that is hours of silent
            # hanging on one stalled connection (observed in practice: 360 calls still unfinished after
            # an hour, and Ctrl-C could not interrupt it because ThreadPoolExecutor.__exit__ waits
            # on the blocked worker threads). 90 s is ~5x a normal judged call, and max_retries=0
            # hands retry policy to our loop alone so the two ladders cannot multiply.
            self._client = anthropic.Anthropic(timeout=90.0, max_retries=0)

    def ask(self, key, system, user, max_tokens=None):
        max_tokens = max_tokens or self.max_tokens
        if key in self.cache:
            with self.lock:
                self.usage["cache_hits"] += 1
            return self.cache[key]
        if self.dry_run:
            return "(dry-run)"
        last_err = None
        for attempt in range(self.max_attempts):
            try:
                kw = {"temperature": 0} if self.send_temperature else {}
                msg = self._client.messages.create(
                    model=self.model, max_tokens=max_tokens,
                    system=system, messages=[{"role": "user", "content": user}], **kw)
                text = "".join(b.text for b in msg.content if b.type == "text").strip()
                with self.lock:
                    self.usage["input"] += msg.usage.input_tokens
                    self.usage["output"] += msg.usage.output_tokens
                    self.usage["calls"] += 1
                    self.cache[key] = text
                    with self.cache_path.open("a") as f:
                        f.write(json.dumps({"key": key, "model": self.model,
                                            "response": text}) + "\n")
                return text
            except Exception as e:
                last_err = e
                status = getattr(e, "status_code", None)
                # Do NOT gate this on self.send_temperature: with parallel workers, another
                # thread may already have cleared it while this request was in flight.
                if status == 400 and "temperature" in str(e):
                    self.send_temperature = False  # retry immediately without it
                    continue
                # CREDENTIALS ARE NOT A TRANSIENT FAULT. A missing/invalid key surfaces with no
                # status_code (the SDK raises "Could not resolve authentication method" before any
                # HTTP call), so it used to fall through to the retry ladder: 8 attempts x backoff
                # x every call = 26 min of sleeping, then a table of 0.000 scores that looked like
                # a catastrophic result instead of an unset env var. Abort the process outright —
                # safe_ask deliberately does NOT catch this, because continuing is never right.
                msg = str(e).lower()
                if status in (401, 403) or "authentication" in msg or "api_key" in msg:
                    sys.exit(f"\nABORT: no usable Anthropic credentials ({e}).\n"
                             f"       Set the key and re-run — cached results are reused, so only "
                             f"the unfinished calls are re-issued:\n"
                             f"         read -s ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY\n")
                if status is not None and 400 <= status < 500 and status != 429:
                    raise RuntimeError(f"Claude call failed, not retryable ({key}): {e}") from e
                # Retryable: 429, 5xx, 529 "overloaded", connection errors. Overload is
                # sustained (not milliseconds), so back off long and jitter so the workers
                # do not retry in lockstep and re-spike the same instant.
                if attempt < self.max_attempts - 1:
                    time.sleep(min(2 ** (attempt + 1), 30) * (0.5 + random.random()))
        raise RuntimeError(
            f"Claude call failed after {self.max_attempts} attempts ({key}): {last_err}")


    def invalidate(self, pred):
        """Drop cached responses matching pred(key, response) and rewrite the cache file.
        Used to re-ask judge calls whose response was truncated/empty."""
        with self.lock:
            dropped = [k for k, v in self.cache.items() if pred(k, v)]
            for k in dropped:
                del self.cache[k]
            if dropped and self.cache_path.exists():
                rows = [r for r in load_jsonl(self.cache_path) if r["key"] not in set(dropped)]
                self.cache_path.write_text(
                    "".join(json.dumps(r) + "\n" for r in rows))
        return len(dropped)


VERDICTS = ("equivalent", "wrong", "gold_problem")


def parse_verdict(raw, allowed=VERDICTS):
    """Tolerant of truncated JSON: the verdict field is emitted first, so recover it even
    when the trailing reason string is cut off by max_tokens.

    `allowed` is the verdict vocabulary. It MUST be passed by callers that use a different one
    (e.g. claude_datacheck's supported/unsupported/contradicted) — otherwise every verdict is
    silently rejected as unparseable, which is exactly how a 240-call audit returned 0% coverage."""
    if not raw or not raw.strip():
        return None, ""
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            v = json.loads(m.group(0))
            if v.get("verdict") in allowed:
                return v["verdict"], str(v.get("reason", ""))
    except json.JSONDecodeError:
        pass
    m = re.search(r'"verdict"\s*:\s*"(\w+)"', raw)
    if m and m.group(1) in allowed:
        r = re.search(r'"reason"\s*:\s*"(.*)', raw, re.DOTALL)
        return m.group(1), (r.group(1).rstrip('"} \n') if r else "")
    return None, raw[:200]


# Below this share of parsed verdicts a system's audited score is reported as unavailable rather
# than computed — missing judgments must never be silently counted as wrong answers.
MIN_JUDGE_COVERAGE = 0.90

FAILED = []          # keys whose call never succeeded, even after all retries
_fail_lock = threading.Lock()


def safe_ask(oracle, key, system, user, max_tokens=None):
    """Never let one exhausted call abort a whole arm. Successful calls are already cached,
    so re-running the same command retries only the stragglers."""
    try:
        return oracle.ask(key, system, user, max_tokens=max_tokens)
    except RuntimeError as e:
        with _fail_lock:
            FAILED.append((key, str(e)[:120]))
        return ""


def run_parallel(fn, items, workers, progress_every=25):
    """Parallel map that REPORTS PROGRESS. `ex.map` returns only when every call is done, so a long
    arm printed nothing for its whole duration and a throttled run was indistinguishable from a hang.
    Results are re-sorted to the input order, so callers that zip against `items` are unaffected."""
    out, done, total = [None] * len(items), 0, len(items)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for fut in as_completed(futs):
            i = futs[fut]
            out[i] = fut.result()
            done += 1
            if done % progress_every == 0 or done == total:
                el = time.time() - t0
                rate = done / el if el else 0
                print(f"    … {done}/{total} judged  ({el/60:.1f} min elapsed, "
                      f"{rate*60:.0f}/min, ~{(total-done)/rate/60:.1f} min left)" if rate else
                      f"    … {done}/{total} judged", flush=True)
    return out


# ── arms ─────────────────────────────────────────────────────────────────────

def score(pred, gold):
    return {"fuzzy": fuzzy_containment(pred, gold), "cont": containment(pred, gold),
            "em": exact_match(pred, gold), "f1": f1_score(pred, gold)}


def arm_answering(arm, oracle, questions, articles, workers):
    """ceiling (gold articles in context) or parametric (no context)."""
    def one(q):
        if arm == "ceiling":
            parts = []
            for i, s in enumerate(q["article_slugs"], 1):
                a = articles[s]
                parts.append(f"Article {i} — {a['title']}:\n{a['context']}")
            user = "\n\n".join(parts) + f"\n\nQuestion: {q['question']}"
            system = CEILING_SYSTEM
        else:
            user, system = f"Question: {q['question']}", PARAMETRIC_SYSTEM
        pred = safe_ask(oracle, f"{arm}:{q['id']}:{oracle.model}", system, user)
        return {"id": q["id"], "question": q["question"], "gold": q["answer"],
                "pred": pred, **score(pred, q["answer"])}

    recs = run_parallel(one, questions, workers)
    n = len(recs)
    summary = {k: sum(r[k] for r in recs) / n for k in ("fuzzy", "cont", "em", "f1")}
    return {"arm": arm, "model": oracle.model, "n": n, "summary": summary, "records": recs}


def load_extra_preds(spec):
    """`NAME:FILE:KEY` -> (NAME, {qid: {"pred", "fuzzy"}}), for auditing any additional system.

    Understands three result formats:
      router_results.json      KEY = a policy label ("chain", "dense@1") or "teacher"
      text_rag_results.json    KEY = a record field ("iter_chain", "oracle_text", "dense@2")
      style_lock_results.json  KEY = an "ARM/DECODING" condition ("C/verbose", "A/short", "text/verbose")
                               — the eval_style_lock grid, used for the fidelity-scale arms
    Predictions carry their own recorded fuzzy where available; otherwise it is recomputed, so the
    FP/FN columns always compare the judge against the SAME lexical score the table reports.
    """
    name, path, key = spec.split(":", 2)
    data = json.loads(Path(path).read_text())
    recs = data["records"]
    out = {}
    for r in recs:
        if "policies" in r:                                   # router_results.json
            src = r.get("teacher") if key == "teacher" else r.get("policies", {}).get(key)
            if not src or src.get("pred") is None:
                continue
            gold, pred = r["answer"], src["pred"]
            fz = src.get("fuzzy", fuzzy_containment(pred, gold))
        elif "preds" in r:                                    # style_lock_results.json
            src = r["preds"].get(key)
            if not src or src.get("pred") is None:
                continue
            gold, pred = r["gold"], src["pred"]
            fz = src.get("fuzzy", fuzzy_containment(pred, gold))
        elif "final_slugs" in r:                              # langgraph_results.json
            # CAREFUL: here `gold` is the article-SLUG LIST and the gold ANSWER is `answer` —
            # the opposite of text_rag_results.json, where `gold` IS the answer string. Falling
            # through to that branch would hand the judge a list of slugs as the correct answer.
            # KEY is ignored (one prediction stream per file); accepted for CLI uniformity.
            if r.get("pred") is None:
                continue
            gold, pred = r["answer"], r["pred"]
            fz = r.get("fuzzy", fuzzy_containment(pred, gold))
        else:                                                 # text_rag_results.json
            if r.get(key) is None:
                continue
            gold, pred = r["gold"], r[key]
            fz = fuzzy_containment(pred, gold)
        out[r["id"]] = {"pred": pred, "fuzzy": fz}
    if not out:
        sys.exit(f"ABORT: --audit-extra {spec!r} matched no predictions. Check the KEY: for a "
                 f"router file use a policy label or 'teacher'; for a text_rag file use a record "
                 f"field such as 'iter_chain'; for a style_lock file use 'ARM/DECODING' such as "
                 f"'C/verbose'.")
    return name, out


def arm_judge_all(oracle, questions, preds_by_system, workers, max_tokens=1024):
    """Blind symmetric audit: judge EVERY question for EVERY system under one protocol.

    Catches both error directions that the round-1 one-sided judge could not:
      false positives  - fuzzy scored 1 but the asserted answer is wrong (verbosity artifact)
      false negatives  - fuzzy scored 0 but the answer is right (England/English)
    The judge is never told which system produced a prediction, so no system can be graded
    more leniently than another.
    """
    qs = {q["id"]: q for q in questions}
    oracle.invalidate(lambda k, v: k.startswith("judge2:") and parse_verdict(v)[0] is None)

    items = [(sysname, qid, p["pred"], p["fuzzy"])
             for sysname, preds in preds_by_system.items() for qid, p in preds.items()]

    def one(item):
        sysname, qid, pred, fuzzy = item
        q = qs[qid]
        user = (f"Question: {q['question']}\n"
                f"Gold answer: {q['answer']}\n"
                f"Answer to grade: {pred}")
        # Blinding guard. The prompt is assembled ONLY from question/gold/pred, so an occurrence of
        # the system name that ALSO occurs in that content is the content's own word, not a leak —
        # a bare `sysname not in user` test false-positives on any system named after a common noun
        # ('teacher' legitimately appears in 2 of the 500 scaled questions; at n=100 it never did).
        # A real leak — a label interpolated into the template by a future refactor — would put the
        # name in `user` but not in `content`, which is exactly what this still catches.
        content = f"{q['question']} {q['answer']} {pred}"
        assert not (sysname in user and sysname not in content), \
            f"blinding violated: {sysname!r} leaked into the prompt"
        raw = safe_ask(oracle, f"judge2:{sysname}:{qid}:{oracle.model}", JUDGE2_SYSTEM, user,
                       max_tokens=max_tokens)
        verdict, reason = parse_verdict(raw)
        verdict = verdict or "unparseable"
        return {"system": sysname, "id": qid, "question": q["question"], "gold": q["answer"],
                "pred": pred, "fuzzy": fuzzy, "verdict": verdict, "reason": reason,
                # the two disagreements between the metric and the judge
                "false_positive": fuzzy == 1.0 and verdict == "wrong",
                "false_negative": fuzzy == 0.0 and verdict == "equivalent"}

    recs = run_parallel(one, items, workers)
    gold_problem = sorted({r["id"] for r in recs if r["verdict"] == "gold_problem"})
    out = {"arm": "judge_all", "model": oracle.model, "n_calls": len(recs),
           "gold_problem_ids": gold_problem, "records": recs, "systems": {}}
    for sysname in preds_by_system:
        rs = [r for r in recs if r["system"] == sysname]
        n = len(rs)          # per-system: the teacher can have fewer records (teacher OOM skips)
        raw = sum(1 for r in rs if r["fuzzy"] == 1.0)
        fp = sum(1 for r in rs if r["false_positive"])
        fn = sum(1 for r in rs if r["false_negative"])
        # ONE authoritative definition of "correct" under audit: the blind judge called it
        # equivalent. (Deriving it as raw-FP+FN instead would disagree whenever a question is
        # both fuzzy-correct and gold-broken, and the cross-tab uses this definition.)
        audited = sum(1 for r in rs if r["verdict"] == "equivalent")
        # COVERAGE GATE. An unjudged call lands as "unparseable", which counts as not-equivalent —
        # so a system whose calls all failed (auth error, sustained overload) scored a confident
        # 0.000 that read like a catastrophic result rather than a broken run. Never let missing
        # verdicts masquerade as wrong answers; report the score as unavailable instead.
        parsed = sum(1 for r in rs if r["verdict"] != "unparseable")
        coverage = parsed / n if n else 0.0
        # dataset-adjusted: drop questions whose gold the judge flagged as broken (for EITHER
        # system), applied identically to both so the comparison stays like-for-like.
        keep = [r for r in rs if r["id"] not in set(gold_problem)]
        adj_ok = sum(1 for r in keep if r["verdict"] == "equivalent")
        out["systems"][sysname] = {
            "raw_fuzzy": raw / n, "false_positives": fp, "false_negatives": fn,
            "coverage": coverage,
            "audited": audited / n if coverage >= MIN_JUDGE_COVERAGE else None,
            "dataset_adjusted": (adj_ok / len(keep) if keep else None)
                                if coverage >= MIN_JUDGE_COVERAGE else None,
            "n_adjusted": len(keep),
            "verdicts": {v: sum(1 for r in rs if r["verdict"] == v)
                         for v in ("equivalent", "wrong", "gold_problem", "unparseable")},
        }
    return out


def arm_nhop_oneshot(oracle, nhop_mh, nhop_text, workers, max_tokens=200):
    """Opus one-shot on the synthetic N-hop chains — does 'you must decompose' survive frontier scale?

    Mirrors scripts/eval_nhop.py's `oneshot` condition exactly so the number is drop-in comparable:
    all gold chain articles concatenated as context + the original nested question, scored with
    `containment` (what eval_nhop uses -- NOT fuzzy). The corpus is fictional, so parametric
    contamination is zero by construction.
    """
    def one(q):
        ctx = "\n\n".join(nhop_text[s] for s in q["chain_slugs"])
        pred = safe_ask(oracle, f"nhop1shot:{q['id']}:{oracle.model}", NHOP_SYSTEM,
                        f"{ctx}\n\nQuestion: {q['question']}", max_tokens=max_tokens)
        return {"id": q["id"], "depth": q["depth"], "question": q["question"],
                "gold": q["answer"], "pred": pred,
                "containment": containment(pred, q["answer"])}

    recs = run_parallel(one, nhop_mh, workers)
    by_depth = {}
    for d in sorted({r["depth"] for r in recs}):
        rs = [r for r in recs if r["depth"] == d]
        by_depth[d] = {"n": len(rs), "oneshot": sum(r["containment"] for r in rs) / len(rs)}
    return {"arm": "nhop_oneshot", "model": oracle.model, "by_depth": by_depth, "records": recs}


def arm_judge(oracle, questions, chain, workers, max_tokens=1024):
    qs = {q["id"]: q for q in questions}
    failures = [(qid, c) for qid, c in chain.items() if c["fuzzy"] == 0.0]
    # A truncated/empty cached verdict is worse than none: re-ask it.
    n = oracle.invalidate(lambda k, v: k.startswith("judge:") and parse_verdict(v)[0] is None)
    if n:
        print(f"  (re-asking {n} cached verdicts that were empty or truncated)")

    def one(item):
        qid, c = item
        q = qs[qid]
        user = (f"Question: {q['question']}\n"
                f"Gold answer: {q['answer']}\n"
                f"System prediction: {c['pred']}\n"
                + (f"(For reference, a strong reader with the source text answered: "
                   f"{c['teacher_pred']})\n" if c.get("teacher_pred") else ""))
        raw = safe_ask(oracle, f"judge:{qid}:{oracle.model}", JUDGE_SYSTEM, user,
                       max_tokens=max_tokens)
        verdict, reason = parse_verdict(raw)
        return {"id": qid, "question": q["question"], "gold": q["answer"],
                "pred": c["pred"], "teacher_pred": c.get("teacher_pred"),
                "verdict": verdict or "unparseable", "reason": reason}

    recs = run_parallel(one, failures, workers)
    counts = {}
    for r in recs:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return {"arm": "judge", "model": oracle.model, "n_failures": len(recs),
            "counts": counts, "records": recs}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multihop", default="data/2wiki_v2/multihop.jsonl")
    ap.add_argument("--articles", default="data/2wiki_v2/articles.jsonl")
    ap.add_argument("--router-results", default="results/chain_fix/R3/router_results.json")
    ap.add_argument("--results-dir", default="results/claude_oracle")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--arm", default="all",
                    choices=["ceiling", "parametric", "judge", "judge_all", "nhop_oneshot", "all"])
    ap.add_argument("--audit-extra", action="append", default=[], metavar="NAME:FILE:KEY",
                    help="audit an additional prediction stream under the same blind protocol, "
                         "e.g. chain_terse:results/short_answer/chain_short/router_results.json:chain "
                         "or ircot_terse:results/short_answer/text_short_fixed/text_rag_results.json:"
                         "iter_chain. Repeatable.")
    ap.add_argument("--nhop-multihop", default="data/synth_chain2/multihop.jsonl")
    ap.add_argument("--nhop-corpus", default="data/synth_chain2/corpus.jsonl")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--expect-chain", type=float, default=0.71,
                    help="guard: recomputed chain fuzzy must match this before any API call")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls; exercises loading/guard/cache/report paths")
    args = ap.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions, articles, chain, chain_fuzzy, teacher_fuzzy = load_inputs(args)

    oracle = Oracle(args.model, out_dir / "cache.jsonl", dry_run=args.dry_run,
                    max_tokens=args.max_tokens)
    arms = (["ceiling", "parametric", "judge_all", "nhop_oneshot"]
            if args.arm == "all" else [args.arm])
    results = {}
    for arm in arms:
        print(f"\n── arm: {arm} ──")
        if arm == "nhop_oneshot":
            mh_path, corpus_path = Path(args.nhop_multihop), Path(args.nhop_corpus)
            if not (mh_path.exists() and corpus_path.exists()):
                print(f"  SKIP: {mh_path} / {corpus_path} not found — pull data/synth_chain2/ "
                      f"by running scripts/build_synth_chain.py first.")
                continue
            nhop_mh = load_jsonl(mh_path)
            nhop_text = {a["passage_id"]: a["context"] for a in load_jsonl(corpus_path)}
            depths = {}
            for q in nhop_mh:
                depths[q["depth"]] = depths.get(q["depth"], 0) + 1
            print(f"  {len(nhop_mh)} chains by depth {dict(sorted(depths.items()))}")
            missing = [s for q in nhop_mh for s in q["chain_slugs"] if s not in nhop_text]
            if missing:
                sys.exit(f"ABORT: chain slugs missing from {corpus_path}: {sorted(set(missing))[:5]}")
            res = arm_nhop_oneshot(oracle, nhop_mh, nhop_text, args.max_workers)
            print(f"  {'depth':>5} {'n':>4} {'Opus 1-shot':>12}   (4B ref: 1-shot / chain)")
            ref = {2: (0.27, 0.60), 3: (0.00, 0.47), 4: (0.00, 0.40)}
            for d, v in res["by_depth"].items():
                r = ref.get(d)
                tail = f"   ({r[0]:.2f} / {r[1]:.2f})" if r else ""
                print(f"  {d:>5} {v['n']:>4} {v['oneshot']:>12.3f}{tail}")
        elif arm == "judge_all":
            ceiling_recs = results.get("ceiling")
            if ceiling_recs is None:
                p = out_dir / "ceiling.json"
                if not p.exists():
                    sys.exit("ABORT: judge_all needs the ceiling arm — run --arm ceiling first.")
                ceiling_recs = json.loads(p.read_text())
                results["ceiling"] = ceiling_recs  # so the cross-tab below still runs
            preds = {
                "chain": {qid: {"pred": c["pred"], "fuzzy": c["fuzzy"]}
                          for qid, c in chain.items()},
                "ceiling": {r["id"]: {"pred": r["pred"], "fuzzy": r["fuzzy"]}
                            for r in ceiling_recs["records"]},
            }
            # The 4B teacher is audited too: it is the reference the headline "chain 0.71 >
            # teacher 0.64" rests on, and both are verbose readers whose containment scores can
            # be inflated by refusals that quote the passage. Auditing only one of them would
            # repeat exactly the asymmetry this arm exists to remove.
            tp = {qid: {"pred": c["teacher_pred"], "fuzzy": c["teacher_fuzzy"]}
                  for qid, c in chain.items()
                  if c.get("teacher_pred") is not None and c.get("teacher_fuzzy") is not None}
            if tp:
                preds["teacher"] = tp
            for spec in args.audit_extra:
                nm, pr = load_extra_preds(spec)
                preds[nm] = pr
                print(f"  + auditing '{nm}' ({len(pr)} predictions from {spec.split(':')[1]})")
            res = arm_judge_all(oracle, questions, preds, args.max_workers)
            print(f"  {'system':<10} {'raw':>6} {'FP':>4} {'FN':>4} {'audited':>8} "
                  f"{'adjusted':>9} (n={res['systems']['chain']['n_adjusted']})")
            for sysname, s in res["systems"].items():
                adj, aud, cov = s["dataset_adjusted"], s["audited"], s["coverage"]
                warn = "" if aud is not None else \
                    f"   ⚠️ only {cov:.0%} of calls returned a verdict — re-run to retry"
                print(f"  {sysname:<10} {s['raw_fuzzy']:>6.3f} {s['false_positives']:>4} "
                      f"{s['false_negatives']:>4} "
                      f"{(f'{aud:.3f}' if aud is not None else 'NO DATA'):>8} "
                      f"{(f'{adj:.3f}' if adj is not None else '   —'):>9}{warn}")
            print(f"  broken golds (excluded from 'adjusted'): {len(res['gold_problem_ids'])}")
            flips = [r for r in res["records"] if r["false_positive"] or r["false_negative"]]
            print(f"  metric/judge disagreements: {len(flips)}")
            for r in flips:
                kind = "FALSE-POS" if r["false_positive"] else "FALSE-NEG"
                print(f"    [{kind}] {r['system']:<8} gold={r['gold']!r} "
                      f"pred={r['pred'][:64]!r}\n{'':16}{r['reason'][:150]}")
        elif arm == "judge":
            res = arm_judge(oracle, questions, chain, args.max_workers)
            for r in res["records"]:
                print(f"  [{r['verdict']:>12}] {r['id']}  gold={r['gold']!r}  "
                      f"pred={r['pred']!r}\n{'':16}{r['reason']}")
            print(f"  verdicts: {res['counts']}")
        else:
            res = arm_answering(arm, oracle, questions, articles, args.max_workers)
            s = res["summary"]
            print(f"  fuzzy={s['fuzzy']:.3f}  contains-EM={s['cont']:.3f}  "
                  f"EM={s['em']:.3f}  F1={s['f1']:.3f}")
        results[arm] = res
        (out_dir / f"{arm}.json").write_text(json.dumps(res, indent=2))

    # ── combined table ────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"CLAUDE ORACLE — {args.model} · {len(questions)} questions")
    print(f"{'condition':<38} {'fuzzy':>6} {'cEM':>6} {'EM':>6} {'F1':>6}")
    for arm in ("ceiling", "parametric"):
        if arm in results:
            s = results[arm]["summary"]
            name = "Claude + gold articles" if arm == "ceiling" else "Claude no context"
            print(f"{name:<38} {s['fuzzy']:>6.3f} {s['cont']:>6.3f} "
                  f"{s['em']:>6.3f} {s['f1']:>6.3f}")
    if teacher_fuzzy is not None:
        print(f"{'4B teacher (gold in context)':<38} {teacher_fuzzy:>6.3f}")
    print(f"{'cartridge chain (ours)':<38} {chain_fuzzy:>6.3f}")
    if "judge_all" in results:
        ja = results["judge_all"]
        print(f"\n{'--- blind symmetric audit (both sides judged) ---':<38}")
        LBL = {"ceiling": "Opus + gold", "chain": "cartridge chain", "teacher": "4B teacher"}
        for sysname in ja["systems"]:
            s = ja["systems"][sysname]
            aud, cov = s["audited"], s["coverage"]
            score = f"{aud:6.3f}" if aud is not None else "NO DATA"
            note = "" if aud is not None else f"; only {cov:.0%} judged"
            n_tot = s["n_adjusted"] + len(ja["gold_problem_ids"])
            print(f"{LBL.get(sysname, sysname) + ', AUDITED':<38} {score}   "
                  f"(raw {s['raw_fuzzy']:.3f}; {s['false_positives']} FP, "
                  f"{s['false_negatives']} FN, n={n_tot}{note})")
        for sysname in ja["systems"]:
            s = ja["systems"][sysname]
            if s["dataset_adjusted"] is None: continue
            print(f"{LBL.get(sysname, sysname) + ', dataset-adj':<38} "
                  f"{s['dataset_adjusted']:>6.3f}   (n={s['n_adjusted']}, "
                  f"{len(ja['gold_problem_ids'])} broken golds dropped)")
    elif "judge" in results:
        eq = results["judge"]["counts"].get("equivalent", 0)
        gp = results["judge"]["counts"].get("gold_problem", 0)
        corrected = chain_fuzzy + eq / len(questions)
        print(f"{'judge-corrected chain (+equivalents)':<38} {corrected:>6.3f}   "
              f"({eq} metric-noise, {gp} gold_problem of "
              f"{results['judge']['n_failures']} failures)")
    # ── cross-tabulation (the actual finding: where the headroom lives) ───────
    if "ceiling" in results and ("judge_all" in results or "judge" in results):
        if "judge_all" in results:
            # audited labels: a system is correct iff the blind judge called it equivalent
            ok = {(r["system"], r["id"]): r["verdict"] == "equivalent"
                  for r in results["judge_all"]["records"]}
            C = {i: ok.get(("ceiling", i), False) for i in chain}
            K = {i: ok.get(("chain", i), False) for i in chain}
            print("\n(cross-tab below uses AUDITED labels from the blind judge)")
        else:
            eqv = {r["id"] for r in results["judge"]["records"] if r["verdict"] == "equivalent"}
            C = {r["id"]: r["fuzzy"] == 1.0 for r in results["ceiling"]["records"]}
            K = {i: (chain[i]["fuzzy"] == 1.0 or i in eqv) for i in chain}
        cells = {(c, k): sum(1 for i in chain if C[i] == c and K[i] == k)
                 for c in (True, False) for k in (True, False)}
        print(f"\nchain (judge-corrected) × Claude+gold ceiling, n={len(chain)}")
        print(f"  both solve                                   {cells[(True, True)]:>3}")
        print(f"  RECOVERABLE HEADROOM  (Opus+gold ✓, ours ✗)  {cells[(True, False)]:>3}")
        print(f"  ours ✓, Opus+gold ✗                          {cells[(False, True)]:>3}")
        print(f"  neither solves (true unanswerable floor)     {cells[(False, False)]:>3}")
        if "parametric" in results:
            P = {r["id"]: r["fuzzy"] == 1.0 for r in results["parametric"]["records"]}
            npar = sum(P.values())
            print(f"\nparametric structure: Opus answers {npar}/{len(chain)} with NO context")
            print(f"  evidence wins (gold ✓, parametric ✗)         "
                  f"{sum(1 for i in chain if C[i] and not P[i]):>3}")
            print(f"  our chain on the {npar} parametric-solvable      "
                  f"{sum(1 for i in chain if P[i] and K[i]):>3}")
            print(f"  our chain on the {len(chain)-npar} NOT parametric-solvable  "
                  f"{sum(1 for i in chain if not P[i] and K[i]):>3}")

    if FAILED:
        print(f"\n!! {len(FAILED)} calls never succeeded (API overload / exhausted "
              f"retries). Successful calls ARE cached — just re-run the same command "
              f"to retry only these.")
        for k, e in FAILED[:5]:
            print(f"     {k}  {e}")
    u = oracle.usage
    print(f"\nAPI usage: {u['calls']} calls ({u['cache_hits']} cache hits) · "
          f"{u['input']:,} in / {u['output']:,} out tokens")


if __name__ == "__main__":
    main()
