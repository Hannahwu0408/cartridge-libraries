"""
evaluate_squad.py — Perplexity + generation eval for SQuAD cartridges.

For each passage in passages.jsonl, evaluates each QA under four conditions with
two complementary metrics and writes results to results/squad/perplexity_results.json:

  • Perplexity of the gold answer span (informative for numeric/date answers).
  • Generation metrics: greedily decode an answer and score EM / token-F1 /
    containment vs the gold (the right metric for open-ended entity/phrase answers,
    where even the teacher has high PPL because it never commits to one phrasing).

Results are bucketed by answer type — `numeric` (answer contains a digit) vs
`entity` (everything else) — with PPL as the headline for numeric and containment
the headline for entity.

Conditions
----------
  no_cartridge     — baseline: model sees only the question (no context)
  correct_cartridge — cartridge trained on this passage's context
  wrong_cartridge  — cartridge from a different article (negative control)
  teacher          — model sees the full passage context in the prompt (upper bound)

Usage
-----
python scripts/evaluate_squad.py \
    --passages-file data/squad/pilot/passages.jsonl \
    --cartridge-dir cartridges/squad \
    --cartridge-len 64 \
    --model-id Qwen/Qwen3-1.7B \
    --results-dir results/squad

# Faster perplexity-only run (old behavior)
python scripts/evaluate_squad.py --no-generate

# Inspect one article, printing generated predictions
python scripts/evaluate_squad.py --filter-title "Beyoncé" --verbose
"""

from __future__ import annotations

import re
import json
import math
import string
import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from cartridge_qwen import (
    get_device, load_model, compute_perplexity, generate_answer, QwenCartridge
)

CONDITIONS = ["no_cartridge", "correct_cartridge", "wrong_cartridge", "teacher"]


# ── Aggregation helpers ────────────────────────────────────────────────────────────

def geo_mean(values: list[float]) -> float | None:
    valid = [v for v in values if not math.isnan(v) and v > 0]
    if not valid:
        return None
    return math.exp(sum(math.log(v) for v in valid) / len(valid))


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _fmt(v: float | None) -> str:
    return f"{v:8.1f}" if v is not None else "     n/a"


def _fmtp(v: float | None) -> str:
    """Format a fraction in [0,1] as a percentage."""
    return f"{100 * v:7.1f}%" if v is not None else "     n/a"


# ── SQuAD-style answer scoring (no existing util in repo) ──────────────────────────

_ARTICLES = {"a", "an", "the"}


def normalize_text(s: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace (SQuAD-style)."""
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def containment(pred: str, gold: str) -> float:
    """1.0 if the normalized gold appears as a substring of the normalized prediction."""
    g = normalize_text(gold)
    p = normalize_text(pred)
    if not g:
        return float("nan")
    return 1.0 if g in p else 0.0


def fuzzy_containment(pred: str, gold: str, threshold: float = 0.7) -> float:
    """1.0 if some word-window of the prediction matches the gold with char-level
    similarity >= threshold (difflib ratio). Credits synthesized-but-misspelled answers
    (e.g. 'Qelmoor Veyl' vs 'Qelmoor Vast') that exact containment scores 0 — the OOV-copy
    degradation composition suffers under diluted attention."""
    from difflib import SequenceMatcher
    g = normalize_text(gold)
    if not g:
        return float("nan")
    pw = normalize_text(pred).split()
    if not pw:
        return 0.0
    n = len(g.split())
    best = 0.0
    for w in range(max(1, n - 1), n + 2):
        for i in range(0, max(1, len(pw) - w + 1)):
            r = SequenceMatcher(None, g, " ".join(pw[i:i + w])).ratio()
            if r > best:
                best = r
    return 1.0 if best >= threshold else 0.0


def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()
    if not pred_toks or not gold_toks:
        return 1.0 if pred_toks == gold_toks else 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def classify_answer(answer: str) -> str:
    """`numeric` if the answer contains a digit (dates/years/counts/%), else `entity`."""
    return "numeric" if re.search(r"\d", answer) else "entity"


def load_passages(path: Path) -> list[dict]:
    passages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            passages.append(json.loads(line))
    return passages


def find_wrong_cartridge_path(
    current_pid: str,
    current_title: str,
    cartridge_dir: Path,
    cartridge_len: int,
) -> Path | None:
    """
    Find a cartridge from a different article to use as the negative control.
    Prefers a cartridge whose article title differs from current_title.
    Returns None if no alternative is found.
    """
    for candidate in sorted(cartridge_dir.iterdir()):
        cart_path = candidate / f"cartridge_{cartridge_len}.pt"
        if not cart_path.exists():
            continue
        # Skip cartridges from the same article (same title slug prefix)
        title_slug = current_title.lower().replace(" ", "_")
        if candidate.name.startswith(title_slug[:10]):
            continue
        # Also skip the correct cartridge itself
        if candidate.name == current_pid:
            continue
        return cart_path
    return None


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SQuAD cartridge perplexity under all conditions"
    )
    parser.add_argument(
        "--passages-file", default="data/squad/pilot/passages.jsonl"
    )
    parser.add_argument(
        "--cartridge-dir", default="cartridges/squad"
    )
    parser.add_argument(
        "--cartridge-len", type=int, default=64
    )
    parser.add_argument(
        "--model-id", default="Qwen/Qwen3-1.7B"
    )
    parser.add_argument(
        "--results-dir", default="results/squad"
    )
    parser.add_argument(
        "--filter-title", default=None,
        help="Only evaluate passages from this article title"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-question perplexity breakdowns"
    )
    parser.add_argument(
        "--no-teacher", action="store_true",
        help="Skip teacher condition (saves time; teacher always wins)"
    )
    parser.add_argument(
        "--no-generate", action="store_true",
        help="Skip generation metrics (EM/F1/containment); perplexity only (old behavior)"
    )
    parser.add_argument(
        "--gen-max-tokens", type=int, default=48,
        help="Max new tokens when greedily decoding answers (default: 48; the model "
             "often emits a short preamble before the answer span, so give it room)"
    )
    parser.add_argument(
        "--max-questions", type=int, default=None,
        help="Cap the number of QAs evaluated per passage (first N). Useful on a large "
             "corpus where the teacher condition re-prefills the whole corpus per question."
    )
    parser.add_argument(
        "--conversational-eval", action="store_true",
        help="Generate answers under the SELF-STUDY chat format (_build_conv_prefix, "
             "no 'Short answer:' cue) instead of the extractive prompt. REQUIRED for "
             "cartridges trained with full-conversation KL — under the extractive "
             "prompt they terminate immediately (out of distribution) and score 0. "
             "Applied to all conditions so teacher/cartridge/no-cartridge stay "
             "comparable; containment scoring handles the full-sentence answers."
    )
    args = parser.parse_args()
    do_generate = not args.no_generate

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    passages = load_passages(Path(args.passages_file))
    if args.filter_title:
        passages = [p for p in passages if p["title"] == args.filter_title]
        print(f"Filtered to {len(passages)} passages from '{args.filter_title}'")

    model, tokenizer = load_model(args.model_id, device)
    cart_dir = Path(args.cartridge_dir)

    all_qa: list[dict] = []  # flat list of per-QA records across all passages

    for entry in passages:
        pid = entry["passage_id"]
        title = entry["title"]
        context = entry["context"]
        qas = entry["qas"]
        if args.max_questions:
            qas = qas[:args.max_questions]

        print(f"\n{'─'*60}")
        print(f"  {pid}  |  {title}")
        print(f"  {len(qas)} QA pairs,  context {len(context)} chars")

        # Load correct cartridge
        cart_path = cart_dir / pid / f"cartridge_{args.cartridge_len}.pt"
        if not cart_path.exists():
            print(f"  SKIP: cartridge not found at {cart_path}")
            print(f"  Run train_squad_cartridges.py first.")
            continue

        correct_cart = QwenCartridge.load(cart_path, device=device)

        # Load wrong cartridge (negative control from a different article)
        wrong_path = find_wrong_cartridge_path(pid, title, cart_dir, args.cartridge_len)
        wrong_cart = QwenCartridge.load(wrong_path, device=device) if wrong_path else None
        if wrong_cart is None:
            print("  NOTE: no wrong-cartridge found (need at least 2 articles trained)")

        # How to invoke each condition: (perplexity kwargs, generation kwargs)
        cond_kwargs = {
            "no_cartridge": {},
            "correct_cartridge": {"cartridge": correct_cart},
            "wrong_cartridge": {"cartridge": wrong_cart},
            "teacher": {"passage": context},
        }

        for qa in qas:
            question = qa["question"]
            raw_ans = qa.get("answers", [])
            answer = raw_ans[0]["text"] if raw_ans else ""
            if not answer:
                continue

            rec: dict = {
                "passage_id": pid,
                "title": title,
                "question": question,
                "answer": answer,
                "answer_type": classify_answer(answer),
            }

            for cond in CONDITIONS:
                kw = cond_kwargs[cond]
                # Conditions that are unavailable this run → NaN / empty.
                unavailable = (
                    (cond == "teacher" and args.no_teacher)
                    or (cond == "wrong_cartridge" and wrong_cart is None)
                )
                if unavailable:
                    rec[f"{cond}_ppl"] = float("nan")
                    rec[f"{cond}_pred"] = ""
                    rec[f"{cond}_em"] = float("nan")
                    rec[f"{cond}_f1"] = float("nan")
                    rec[f"{cond}_cont"] = float("nan")
                    continue

                rec[f"{cond}_ppl"] = compute_perplexity(
                    model, tokenizer, question, answer, device=device, **kw
                )
                if do_generate:
                    pred = generate_answer(
                        model, tokenizer, question, device=device,
                        max_new_tokens=args.gen_max_tokens,
                        conversational=args.conversational_eval, **kw
                    )
                    rec[f"{cond}_pred"] = pred
                    rec[f"{cond}_em"] = exact_match(pred, answer)
                    rec[f"{cond}_f1"] = f1_score(pred, answer)
                    rec[f"{cond}_cont"] = containment(pred, answer)
                else:
                    rec[f"{cond}_pred"] = ""
                    rec[f"{cond}_em"] = float("nan")
                    rec[f"{cond}_f1"] = float("nan")
                    rec[f"{cond}_cont"] = float("nan")

            all_qa.append(rec)

            if args.verbose:
                print(f"    [{rec['answer_type']}] Q: {question[:55]}  (gold: {answer})")
                for cond in CONDITIONS:
                    line = f"      {cond:<18} ppl={rec[f'{cond}_ppl']:.1f}"
                    if do_generate:
                        line += (f"  cont={rec[f'{cond}_cont']:.0f}"
                                 f"  f1={rec[f'{cond}_f1']:.2f}  pred={rec[f'{cond}_pred'][:40]!r}")
                    print(line)

            # The teacher condition re-prefills the whole corpus per question; release
            # cached blocks each step so a large corpus doesn't fragment over many QAs.
            if device == "cuda":
                torch.cuda.empty_cache()

    # ── Aggregate + report, bucketed by answer type ────────────────────────────────

    def summarize(records: list[dict]) -> dict:
        """Per-condition geo-mean PPL and mean EM/F1/containment over `records`."""
        out = {}
        for cond in CONDITIONS:
            out[cond] = {
                "ppl": geo_mean([r[f"{cond}_ppl"] for r in records]),
                "em": mean([r[f"{cond}_em"] for r in records]),
                "f1": mean([r[f"{cond}_f1"] for r in records]),
                "cont": mean([r[f"{cond}_cont"] for r in records]),
                "n": len(records),
            }
        return out

    buckets = {
        "numeric": [r for r in all_qa if r["answer_type"] == "numeric"],
        "entity": [r for r in all_qa if r["answer_type"] == "entity"],
        "all": all_qa,
    }
    summary = {b: summarize(recs) for b, recs in buckets.items()}

    for bucket in ("numeric", "entity", "all"):
        recs = buckets[bucket]
        headline = "PPL↓" if bucket == "numeric" else "containment↑"
        print(f"\n{'='*72}")
        print(f"  {bucket.upper()} answers   (n={len(recs)})   headline metric: {headline}")
        print(f"  {'Condition':<20}{'PPL':>10}{'EM':>9}{'F1':>9}{'Contain':>9}")
        print("  " + "-" * 57)
        for cond in CONDITIONS:
            s = summary[bucket][cond]
            print(f"  {cond:<20}{_fmt(s['ppl'])}"
                  f"{_fmtp(s['em'])}{_fmtp(s['f1'])}{_fmtp(s['cont'])}")

    # Headline PASS/FAIL checks
    print(f"\n{'='*72}")
    num = summary["numeric"]
    ent = summary["entity"]
    if num["no_cartridge"]["ppl"] and num["correct_cartridge"]["ppl"]:
        ok = num["correct_cartridge"]["ppl"] < num["no_cartridge"]["ppl"]
        print(f"  NUMERIC : correct PPL {num['correct_cartridge']['ppl']:.1f} "
              f"vs no-cart {num['no_cartridge']['ppl']:.1f}  → {'PASS' if ok else 'WARN'}")
    if do_generate and ent["no_cartridge"]["cont"] is not None:
        cc = ent["correct_cartridge"]["cont"]
        nc = ent["no_cartridge"]["cont"]
        tc = ent["teacher"]["cont"]
        ok = cc is not None and nc is not None and cc > nc
        print(f"  ENTITY  : correct containment {_fmtp(cc)} vs no-cart {_fmtp(nc)} "
              f"→ {'PASS' if ok else 'WARN'}   (teacher ceiling {_fmtp(tc)})")

    # ── Write results JSON ─────────────────────────────────────────────────────────
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "perplexity_results.json"
    output = {
        "config": {
            "model_id": args.model_id,
            "cartridge_len": args.cartridge_len,
            "passages_file": str(args.passages_file),
            "generate": do_generate,
            "gen_max_tokens": args.gen_max_tokens,
        },
        "summary_by_answer_type": summary,
        # Every QA grouped by passage (passages in file order, questions in their
        # original SQuAD order) so you can see how different question types behave
        # within a single passage.
        "all_qa_results": all_qa,
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults written → {out_path}")


if __name__ == "__main__":
    main()
