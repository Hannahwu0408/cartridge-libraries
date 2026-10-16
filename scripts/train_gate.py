"""
train_gate.py — Phase 11 S1: train a RelevanceGate ONLY (cartridges frozen).

The gnorm composition gate weights cartridges by content-blind logsumexp(q·k); this trains a
small shared RelevanceGate (per-layer low-rank Pq/Pk) that adds a learned relevance bias so the
answer-relevant cartridge is up-weighted and distractors are down-weighted. Cartridges receive NO
gradient — only the gate trains — so there is zero cross-cartridge gradient interference (sidesteps
the "distractors hurt the good cartridge" caveat) and the gate is a SHARED function that should
generalize to unseen cartridges (the continual-learning property; test with --held-out-ids).

Two training signals (mix freely):
  • SINGLE (`--conversations-file`): cartridge X's own conv → compose [X + K distractors], teacher =
    X's single article. Teaches "weight the ONE relevant cartridge" (SELECTION weighting).
  • BRIDGE / two-gold (`--bridge-conversations`): a bridge conv over pair (A,B) → compose
    [A, B, + K distractors], teacher = BOTH articles in context. Teaches "weight BOTH golds high,
    distractors low" (SYNTHESIS weighting) — the signal single-cartridge training can't provide.
Cartridges get NO gradient in either mode (only the gate trains). Evaluate with
evaluate_composition.py --compose-mode gnorm --gate <gate.pt> (scaling + selection + gate weights).

Usage
-----
python scripts/train_gate.py \
    --passages-file data/synth/articles_mm.jsonl \
    --bridge-conversations data/synth/bridge_all.jsonl \
    --cartridge-dir cartridges/synth_joint2 --cartridge-len 64 \
    --model-id Qwen/Qwen3-4B --out cartridges/synth_joint2/gate_bridge.pt \
    --epochs 6 --lr 1e-3 --rank 32 --p-iso 0.1 --max-distractors 4
"""
from __future__ import annotations

import json
import random
import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cartridge_qwen import get_device, load_model, QwenCartridge, RelevanceGate  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="Train a RelevanceGate only (cartridges frozen)")
    ap.add_argument("--passages-file", required=True)
    ap.add_argument("--conversations-file", default=None,
                    help="single-cartridge self-study convs (SELECTION weighting)")
    ap.add_argument("--bridge-conversations", default=None,
                    help="two-gold bridge convs from generate_bridge_data.py (SYNTHESIS weighting)")
    ap.add_argument("--cartridge-dir", required=True, help="FROZEN cartridge pool")
    ap.add_argument("--cartridge-len", type=int, default=64)
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", required=True, help="path to save gate.pt")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p-iso", type=float, default=0.1,
                    help="prob a step shows ONLY X (no distractors); low so the gate sees clutter")
    ap.add_argument("--max-distractors", type=int, default=5)
    ap.add_argument("--max-answer-tokens", type=int, default=48,
                    help="cap the supervised answer window (bounds the retained student graph → "
                         "memory); the first ~N tokens carry the weighting signal")
    ap.add_argument("--max-question-tokens", type=int, default=96,
                    help="drop convs whose question exceeds N tokens up front (malformed "
                         "long-question convs from JSON-parse fallback spike memory); 0 = keep all")
    ap.add_argument("--max-single-per-cart", type=int, default=None,
                    help="subsample single-cartridge convs per cartridge (2Wiki self-study has "
                         "~3k/article → cap for a tractable epoch); None = keep all")
    ap.add_argument("--held-out-ids", default=None,
                    help="comma-sep passage_ids to EXCLUDE from gate training (generalization test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Device: {device}")
    model, tokenizer = load_model(args.model_id, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    if not args.conversations_file and not args.bridge_conversations:
        ap.error("provide --conversations-file and/or --bridge-conversations")
    passages = {p["passage_id"]: p["context"] for p in load_jsonl(args.passages_file)}

    def has_cart(pid):
        return (Path(args.cartridge_dir) / pid / f"cartridge_{args.cartridge_len}.pt").exists()

    convs_by_pid: dict[str, list[dict]] = {}
    if args.conversations_file:
        for rec in load_jsonl(args.conversations_file):
            if rec["passage_id"] in passages:
                convs_by_pid.setdefault(rec["passage_id"], []).append(rec)
    bridge_convs = []
    if args.bridge_conversations:
        for rec in load_jsonl(args.bridge_conversations):
            a, b = rec.get("passage_ids", [None, None])[:2]
            if a in passages and b in passages and has_cart(a) and has_cart(b):
                bridge_convs.append(rec)

    # Drop convs whose question is too long (a malformed long-question conv makes the retained
    # 36-layer student graph spike GPU memory). Cheap one-time filter; the answer cap can't help
    # here since it only trims the answer, not the question.
    if args.max_question_tokens and args.max_question_tokens > 0:
        def q_ok(rec):
            return len(tokenizer.encode(rec.get("question", ""), add_special_tokens=False)) \
                <= args.max_question_tokens
        n_bridge0 = len(bridge_convs)
        bridge_convs = [r for r in bridge_convs if q_ok(r)]
        n_single0 = sum(len(v) for v in convs_by_pid.values())
        convs_by_pid = {p: [c for c in cs if q_ok(c)] for p, cs in convs_by_pid.items()}
        n_single1 = sum(len(v) for v in convs_by_pid.values())
        print(f"Dropped long-question convs (>{args.max_question_tokens} tok): "
              f"bridge {n_bridge0 - len(bridge_convs)}/{n_bridge0}, single {n_single0 - n_single1}/{n_single0}")

    if args.max_single_per_cart:
        convs_by_pid = {p: (random.sample(cs, args.max_single_per_cart)
                            if len(cs) > args.max_single_per_cart else cs)
                        for p, cs in convs_by_pid.items()}
        print(f"Capped single convs to {args.max_single_per_cart}/cart → "
              f"{sum(len(v) for v in convs_by_pid.values())} total")

    held = {s.strip() for s in args.held_out_ids.split(",")} if args.held_out_ids else set()
    # The pool = every cartridge referenced by any conv (single OR bridge) that exists on disk.
    referenced = set(convs_by_pid) | {p for r in bridge_convs for p in r["passage_ids"]}
    all_pids = [p for p in passages if p in referenced and has_cart(p)]
    train_pids = [p for p in all_pids if p not in held]  # single-conv targets excl. held-out
    print(f"Pool: {len(all_pids)} carts ({len(train_pids)} single-train, {len(held)} held out) | "
          f"single convs: {sum(len(v) for v in convs_by_pid.values())} | bridge convs: {len(bridge_convs)}")

    # Load the WHOLE pool frozen (X + distractors are all sampled from here at inference too).
    carts: dict[str, QwenCartridge] = {}
    for pid in all_pids:
        c = QwenCartridge.load(Path(args.cartridge_dir) / pid / f"cartridge_{args.cartridge_len}.pt",
                               device=device)
        for p in c.parameters():
            p.requires_grad_(False)
        c.eval()
        carts[pid] = c

    ref = carts[all_pids[0]]
    gate = RelevanceGate(ref.num_layers, ref.head_dim, rank=args.rank).to(device)
    gate.train()
    opt = torch.optim.Adam(gate.parameters(), lr=args.lr)
    print(f"Gate: {sum(p.numel() for p in gate.parameters())/1e3:.0f}K params "
          f"(rank={args.rank}, layers={ref.num_layers})")

    cart_len = args.cartridge_len

    def sample_distractors(exclude):
        """K frozen distractors from the pool, excluding the gold(s)."""
        pool = [p for p in all_pids if p not in exclude]
        if not pool:
            return None
        k = random.randint(1, min(args.max_distractors, len(pool)))
        return [carts[o] for o in random.sample(pool, k)]

    # Bridge steps whose golds are BOTH trainable (a held-out gold is never a training target).
    bridge_train = [r for r in bridge_convs if not (set(r["passage_ids"]) & held)]
    cuda = device == "cuda"
    if cuda:
        base_mem = torch.cuda.memory_allocated() / 1e9
        print(f"baseline GPU mem after load: {base_mem:.2f} GB")
    for epoch in range(args.epochs):
        steps = [("single", pid, conv) for pid in train_pids for conv in convs_by_pid[pid]]
        steps += [("bridge", None, r) for r in bridge_train]
        random.shuffle(steps)
        total, n, skipped, oom = 0.0, 0, 0, 0
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        for si, (kind, pid, conv) in enumerate(steps):
            try:
                if kind == "single":
                    if random.random() < args.p_iso:
                        continue  # gate only learns from crowded (distractor) steps
                    distractors = sample_distractors({pid})
                    if not distractors:
                        continue
                    loss = QwenCartridge._distill_loss_for_conv(
                        model, tokenizer, carts[pid], conv, passages[pid], cart_len, device,
                        full_conversation_kl=False, max_answer_tokens=args.max_answer_tokens,
                        distractors=distractors, compose_geom="gnorm", gate=gate,
                    )
                else:  # bridge / two-gold
                    a, b = conv["passage_ids"][:2]
                    distractors = (sample_distractors({a, b})
                                   if random.random() > args.p_iso else None)
                    combined = f"Passage 1:\n{passages[a].strip()}\n\nPassage 2:\n{passages[b].strip()}"
                    loss = QwenCartridge._distill_loss_for_conv(
                        model, tokenizer, carts[a], conv, combined, cart_len, device,
                        full_conversation_kl=False, max_answer_tokens=args.max_answer_tokens,
                        distractors=distractors, compose_geom="gnorm", gate=gate,
                        gold_cartridges=[carts[a], carts[b]],
                    )
                if loss is None:
                    skipped += 1
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
                opt.step()
                total += loss.item()
                n += 1
                del loss
            except torch.cuda.OutOfMemoryError:
                oom += 1
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if oom <= 3:
                    print(f"  OOM at step {si} ({kind}); allocated="
                          f"{torch.cuda.memory_allocated()/1e9:.2f} GB — skipped")
            if cuda and si % 200 == 0:
                print(f"  step {si}/{len(steps)}  alloc={torch.cuda.memory_allocated()/1e9:.2f} "
                      f"peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        peak = f" peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB" if cuda else ""
        print(f"epoch {epoch+1}/{args.epochs}  loss={total/max(n,1):.4f}  "
              f"steps={n} skipped={skipped} oom={oom}{peak}")

    gate.eval()
    gate.save(args.out)
    print(f"Saved gate → {args.out}")


if __name__ == "__main__":
    main()
