"""
generate_bridge_data.py — fresh TWO-GOLD (synthesis) conversations for gate/pair training.

For each gold pair (A, B) in a multihop file, put BOTH articles in context and generate questions
answerable ONLY from both (a fact from A chained with a fact from B), then answer them greedily with
both articles in context. These are the co-attention training examples that per-article self-study
never produces — used to teach a RelevanceGate (Phase 11 S1b) or the cartridges (S2) that TWO
cartridges should both be up-weighted while distractors stay low.

Output — one record per bridge conversation:
    {"passage_ids": [A_slug, B_slug], "seed_type": "bridge", "question": "...", "answer": "..."}

Usage
-----
python scripts/generate_bridge_data.py \
    --multihop-file data/synth/multihop.jsonl --articles-file data/synth/articles_mm.jsonl \
    --model-id Qwen/Qwen3-4B --per-pair 60 --out-file data/synth/bridge_all.jsonl
"""
from __future__ import annotations

import json
import random
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cartridge_qwen import load_model, _build_conv_prefix  # noqa: E402
from generate_selfstudy_data import _generate_one, _generate_batch, _parse_items  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def _bridge_prompt(n: int) -> str:
    return (
        f"You are given two passages (Passage 1 and Passage 2 below). Generate {n} diverse questions "
        f"that can ONLY be answered by COMBINING a fact from Passage 1 with a fact from Passage 2 — "
        f"each question must require information from BOTH passages and be UNANSWERABLE from either "
        f"one alone (e.g., ask about how an entity named in Passage 1 relates, via a shared entity, "
        f"to a specific fact stated only in Passage 2). Include the specific names/entities so it is "
        f"clear what is asked. Do NOT ask anything answerable from a single passage. "
        f"Output ONLY a JSON array of exactly {n} strings, no other text, no markdown fences.")


def main():
    ap = argparse.ArgumentParser(description="Generate two-gold (bridge) synthesis conversations")
    ap.add_argument("--multihop-file", required=True, help="pairs with article_slugs [A,B]")
    ap.add_argument("--articles-file", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--question-model", default=None, help="optional stronger M_Q (decoupled)")
    ap.add_argument("--per-pair", type=int, default=60)
    ap.add_argument("--n-per-call", type=int, default=20)
    ap.add_argument("--q-max-tokens", type=int, default=768)
    ap.add_argument("--a-max-tokens", type=int, default=160)
    ap.add_argument("--q-temperature", type=float, default=0.7)
    ap.add_argument("--answer-batch-size", type=int, default=4)
    ap.add_argument("--out-file", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    a_model, tokenizer = load_model(args.model_id, device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.question_model and args.question_model != args.model_id:
        q_model, q_tok = load_model(args.question_model, device)
        if q_tok.pad_token is None:
            q_tok.pad_token = q_tok.eos_token
    else:
        q_model, q_tok = a_model, tokenizer

    ctx = {a["passage_id"]: a["context"] for a in load_jsonl(args.articles_file)}
    pairs = []
    for q in load_jsonl(args.multihop_file):
        sl = q.get("article_slugs", [])
        if len(sl) == 2 and sl[0] in ctx and sl[1] in ctx:
            pairs.append((sl[0], sl[1]))
    pairs = list(dict.fromkeys(pairs))   # dedup, keep order
    print(f"{len(pairs)} gold pairs from {args.multihop_file}")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (a, b) in enumerate(pairs):
            combined = f"Passage 1:\n{ctx[a].strip()}\n\nPassage 2:\n{ctx[b].strip()}"
            print(f"[{i+1}/{len(pairs)}] {a} + {b}")
            seen, made, rounds = set(), 0, 0
            while made < args.per_pair and rounds < args.per_pair:
                rounds += 1
                q_prefix = _build_conv_prefix(_bridge_prompt(args.n_per_call), combined, q_tok)
                raw = _generate_one(q_model, q_tok, q_prefix, args.q_max_tokens,
                                    args.q_temperature, device)
                qs = [x for x in _parse_items(raw)
                      if x.lower() not in seen and not seen.add(x.lower())][: args.per_pair - made]
                if not qs:
                    continue
                for bi in range(0, len(qs), args.answer_batch_size):
                    chunk = qs[bi:bi + args.answer_batch_size]
                    prefixes = [_build_conv_prefix(x, combined, tokenizer) for x in chunk]
                    answers = _generate_batch(a_model, tokenizer, prefixes, args.a_max_tokens, 0.0, device)
                    for x, ans in zip(chunk, answers):
                        if not ans:
                            continue
                        f.write(json.dumps({"passage_ids": [a, b], "seed_type": "bridge",
                                            "question": x, "answer": ans}, ensure_ascii=False) + "\n")
                        made += 1
                        n_written += 1
                f.flush()
            print(f"  wrote {made} ({rounds} calls)")
    print(f"\nWrote {n_written} bridge conversations → {out_path}")


if __name__ == "__main__":
    main()
