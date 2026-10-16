"""
generate_relational_selfstudy.py — 12-I.4: relational-BINDING self-study for the multi-subject-A cartridge.

Tests whether KV-level intermediate-disambiguation is LEARNABLE under competition. Candidate-3 (relational
self-study framing) was null on SINGLE-subject A — nothing to disambiguate. In multi-subject A there are 3
mentee→master rows, so training cart_A densely + CONTRASTIVELY on the hop-1 binding in the exact bridge
framing ("Who is the mentor of <mentee>?" → <master>) gives a discriminative target that MIGHT make the
correct binding dominate under composition (fewer wrong-intermediate errors).

NOTE probe_A is already 1.00 (cart_A resolves the binding in isolation) — this is not a fidelity fix; it
tests robustness under compositional interference. Fully modular: uses only A's own article text.

Emits ONLY the relational convs (passage_id = E1 slug); mix with base self-study downstream (budget-matched).
Reuses load_model / _build_conv_prefix (cartridge_qwen) + _generate_batch (generate_selfstudy_data).

Usage:
  python scripts/generate_relational_selfstudy.py \
      --articles-file data/synth_amulti_fic/articles_mm.jsonl \
      --multihop-file data/synth_amulti_fic/multihop.jsonl \
      --out-file data/synth_amulti_fic/selfstudy_relational.jsonl --per-mentee 15
"""
from __future__ import annotations

import re
import sys
import json
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cartridge_qwen import load_model, _build_conv_prefix          # noqa: E402
from generate_selfstudy_data import _generate_batch                # noqa: E402

# (mentee, master) rows in a multi-A article
PAIR = re.compile(
    r"([A-Z][\w'’.-]+(?: [A-Z][\w'’.-]+)*) was apprenticed to the master artisan "
    r"([A-Z][\w'’.-]+(?: [A-Z][\w'’.-]+)*),")

# bridge-framed hop-1 relational questions ({m} = mentee) — the framings the bridge query uses
TEMPLATES = [
    "Who was the mentor of {m}?",
    "Who trained {m}?",
    "Under whom did {m} apprentice?",
    "Whom did {m} study under?",
    "Who was {m}'s master?",
    "Who was the teacher of {m}?",
    "{m} was apprenticed to whom?",
    "Name the master who trained {m}.",
    "Who mentored {m}?",
    "{m} studied under which master?",
    "Who was the master artisan that trained {m}?",
    "To whom was {m} apprenticed?",
]


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def build_questions(context: str, per_mentee: int) -> list[str]:
    """per_mentee bridge-framed Qs for EACH mentee in the article (contrastive across all rows)."""
    qs = []
    for mentee, _master in PAIR.findall(context):
        for k in range(per_mentee):
            qs.append(TEMPLATES[k % len(TEMPLATES)].format(m=mentee))
    return qs


def main():
    ap = argparse.ArgumentParser(description="relational-binding self-study for multi-A (12-I.4)")
    ap.add_argument("--articles-file", required=True)
    ap.add_argument("--multihop-file", required=True)
    ap.add_argument("--out-file", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--per-mentee", type=int, default=15, help="relational Qs per mentee (cycles templates)")
    ap.add_argument("--a-max-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    model, tok = load_model(args.model_id, device)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    arts = {a["passage_id"]: a["context"] for a in load_jsonl(args.articles_file)}
    e1_slugs = [q["article_slugs"][0] for q in load_jsonl(args.multihop_file)]

    n = 0
    with open(args.out_file, "w", encoding="utf-8") as f:
        for i, e1 in enumerate(e1_slugs):
            ctx = arts[e1]
            qs = build_questions(ctx, args.per_mentee)
            for b in range(0, len(qs), args.batch_size):
                chunk = qs[b:b + args.batch_size]
                prefixes = [_build_conv_prefix(q, ctx, tok) for q in chunk]
                answers = _generate_batch(model, tok, prefixes, args.a_max_tokens, 0.0, device)  # greedy
                for q, a in zip(chunk, answers):
                    if not a:
                        continue
                    f.write(json.dumps({"passage_id": e1, "seed_type": "mentor_binding",
                                        "question": q, "answer": a}, ensure_ascii=False) + "\n")
                    n += 1
            f.flush()
            print(f"[{i+1}/{len(e1_slugs)}] {e1}: {len(qs)} Qs")

    print(f"\nWrote {n} relational-binding convs → {args.out_file}")


if __name__ == "__main__":
    main()
