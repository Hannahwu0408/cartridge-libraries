"""
train_squad_cartridges.py — Train one QwenCartridge per SQuAD passage.

For each passage in passages.jsonl:
  1. Initialize cartridge from passage prefix (extract KV from frozen model).
  2. SELF-STUDY context distillation on the passage's conversations
     (from generate_selfstudy_data.py).
  3. Save cartridge to cartridges/squad/{passage_id}/cartridge_{len}.pt.

Usage
-----
python scripts/train_squad_cartridges.py \
    --passages-file data/squad/pilot/passages.jsonl \
    --conversations-file data/squad/pilot/selfstudy.jsonl \
    --cartridge-dir cartridges/squad \
    --cartridge-len 64 \
    --epochs 10 \
    --model-id Qwen/Qwen3-1.7B

# Train only specific passages (for debugging):
python scripts/train_squad_cartridges.py \
    --passage-ids beyonc_000 \
    --epochs 5
"""

from __future__ import annotations

import json
import random
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from cartridge_qwen import (
    get_device, load_model, QwenCartridge, build_teacher_corpus_cache
)


def load_passages(passages_file: str | Path) -> list[dict]:
    path = Path(passages_file)
    if not path.exists():
        raise FileNotFoundError(
            f"passages.jsonl not found at {path}\n"
            "Run: python scripts/prepare_squad.py first."
        )
    passages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            passages.append(json.loads(line))
    return passages


def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen3 cartridges on SQuAD passages via context distillation"
    )
    parser.add_argument(
        "--passages-file", default="data/squad/pilot/passages.jsonl",
        help="Path to passages.jsonl from prepare_squad.py"
    )
    parser.add_argument(
        "--cartridge-dir", default="cartridges/squad",
        help="Output root for cartridge .pt files"
    )
    parser.add_argument(
        "--cartridge-len", type=int, default=64,
        help="KV prefix tokens per cartridge (default: 64)"
    )
    parser.add_argument(
        "--conversations-file", default="data/squad/pilot/selfstudy.jsonl",
        help="SELF-STUDY conversations from generate_selfstudy_data.py "
             "(default: data/squad/pilot/selfstudy.jsonl)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Context distillation epochs per cartridge (default: 10)"
    )
    parser.add_argument(
        "--steps-per-epoch", type=int, default=40,
        help="Distillation steps per epoch (default: 40; raised to one pass over "
             "all conversations if a passage has more)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4,
        help="Learning rate for cartridge optimizer (default: 5e-4)"
    )
    parser.add_argument(
        "--model-id", default="Qwen/Qwen3-1.7B",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--passage-ids", default=None,
        help="Comma-separated passage_ids to train (default: all passages)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip passages whose cartridge .pt file already exists"
    )
    parser.add_argument(
        "--checkpoint-epochs", default=None,
        help="Comma-separated epoch numbers to also save (e.g. '1,3,5'). Each writes "
             "to a sibling dir <cartridge-dir>_ep<N>/, so evaluate_squad.py loads it "
             "unchanged. When set, --epochs is forced to max(list)."
    )
    parser.add_argument(
        "--no-freeze-sink", action="store_true",
        help="Train the cartridge attention sink (slot-0 K/V) instead of freezing it "
             "at init. Ablation lever: train_distillation defaults to freeze_sink=True."
    )
    parser.add_argument(
        "--full-conversation-kl", action="store_true",
        help="EXPERIMENTAL — known to collapse generation to newlines on article-scale "
             "corpora (ablation: full-conv KL fits chat-formatting tokens, not the answer, "
             "so the cartridge emits '\\n\\n' forever). Distill KL over the whole shared "
             "conversation span instead of the answer tokens only. Default: answer-only."
    )
    parser.add_argument(
        "--cache-teacher", action="store_true",
        help="Precompute the teacher's corpus-block KV once per passage and reuse it "
             "across steps (forward only question+answer each step). Needed for large "
             "(~20k-token) corpora; harmless but unnecessary for short passages."
    )
    parser.add_argument(
        "--no-reuse", action="store_true",
        help="Each epoch is one shuffled pass over ALL unique conversations (paper "
             "one-epoch-no-reuse) instead of sampling with replacement. Use with a large "
             "unique self-study set."
    )
    parser.add_argument(
        "--compose-aware", action="store_true",
        help="Composition-aware fine-tuning: load each cartridge from --init-from and "
             "fine-tune it with K∈{0,1,2} FROZEN distractor cartridges present in the "
             "student forward (random position), so it stays sharp/findable when composed."
    )
    parser.add_argument(
        "--init-from", default=None,
        help="Dir of pre-trained cartridges to load as init + distractor bank for "
             "--compose-aware (defaults to --cartridge-dir)."
    )
    parser.add_argument(
        "--distractor-pool-size", type=int, default=16,
        help="Cap on the number of candidate distractors per target (0 = all). "
             "Randomly subsampled from the bank."
    )
    parser.add_argument(
        "--max-distractors", type=int, default=2,
        help="Max K distractor cartridges present per step (compose-aware). Use 1 for "
             "pairwise composition / tight-VRAM GPUs (matches A+B inference, less memory)."
    )
    parser.add_argument(
        "--compose-geom", default="concat", choices=["concat", "gnorm"],
        help="Composition geometry to train for (compose-aware): concat = distinct "
             "re-RoPE'd positions; gnorm = overlapping in-distribution + joint-softmax "
             "(order-invariant). Must match the --compose-mode used at eval."
    )
    parser.add_argument(
        "--compose-shared-sink", action="store_true",
        help="gnorm compose-aware only: train WITH one shared neutral sink (strip each "
             "cartridge's slot-0), so cartridges are in-distribution for the shared-sink "
             "fix that unflattens the k≥3 gate. Eval with --gnorm-shared-sink."
    )
    args = parser.parse_args()

    checkpoint_epochs = None
    if args.checkpoint_epochs:
        checkpoint_epochs = {int(x) for x in args.checkpoint_epochs.split(",")}
        args.epochs = max(checkpoint_epochs)
        print(f"Checkpointing at epochs {sorted(checkpoint_epochs)}; epochs set to {args.epochs}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    passages = load_passages(args.passages_file)
    print(f"Loaded {len(passages)} passages from {args.passages_file}")

    # Load SELF-STUDY conversations and group them by passage_id
    conv_path = Path(args.conversations_file)
    if not conv_path.exists():
        raise FileNotFoundError(
            f"conversations file not found at {conv_path}\n"
            "Run: python scripts/generate_selfstudy_data.py first."
        )
    convs_by_pid: dict[str, list[dict]] = {}
    for line in conv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            convs_by_pid.setdefault(rec["passage_id"], []).append(rec)
    print(f"Loaded {sum(len(v) for v in convs_by_pid.values())} conversations "
          f"for {len(convs_by_pid)} passages from {conv_path}")

    # Filter to specific passage_ids if requested
    if args.passage_ids:
        wanted = set(p.strip() for p in args.passage_ids.split(","))
        passages = [p for p in passages if p["passage_id"] in wanted]
        print(f"Filtered to {len(passages)} passage(s): {args.passage_ids}")

    # Load model once (shared across all cartridge trainings)
    model, tokenizer = load_model(args.model_id, device)

    # Composition-aware: load the pre-trained cartridges once as a FROZEN distractor bank
    # (and the per-target init). Each target is fine-tuned with random neighbors present.
    init_dir = Path(args.init_from or args.cartridge_dir)
    distractor_bank: dict[str, "QwenCartridge"] = {}
    if args.compose_aware:
        for entry in load_passages(args.passages_file):
            p = init_dir / entry["passage_id"] / f"cartridge_{args.cartridge_len}.pt"
            if p.exists():
                # Keep the bank on CPU; StackedCartridgeCache moves only the ≤K sampled
                # distractors to GPU per step, so 40 cartridges don't sit in VRAM.
                c = QwenCartridge.load(p, device="cpu")
                for prm in c.parameters():
                    prm.requires_grad_(False)
                distractor_bank[entry["passage_id"]] = c
        print(f"Composition-aware: loaded {len(distractor_bank)} cartridges as the frozen "
              f"distractor bank (on CPU) from {init_dir}")

    cart_dir = Path(args.cartridge_dir)
    n_trained = 0
    n_skipped = 0

    for i, entry in enumerate(passages):
        pid = entry["passage_id"]
        title = entry["title"]
        context = entry["context"]
        conversations = convs_by_pid.get(pid, [])

        out_path = cart_dir / pid / f"cartridge_{args.cartridge_len}.pt"

        if args.skip_existing and out_path.exists():
            print(f"\n[{i+1}/{len(passages)}] {pid}  — SKIP (exists)")
            n_skipped += 1
            continue

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(passages)}] {pid}  |  {title}")
        print(f"  Context length: {len(context)} chars, {len(conversations)} conversations")
        print(f"  Output: {out_path}")

        if len(conversations) == 0:
            print("  SKIP: no conversations (run generate_selfstudy_data.py for this passage).")
            n_skipped += 1
            continue

        # Step 1: initialize cartridge. Composition-aware fine-tuning loads the pre-trained
        # cartridge as init (+ builds this target's frozen distractor pool = the bank minus
        # itself); otherwise init fresh from the passage prefix.
        distractor_pool = None
        if args.compose_aware:
            init_path = init_dir / pid / f"cartridge_{args.cartridge_len}.pt"
            if not init_path.exists():
                print(f"  SKIP: no init cartridge at {init_path} (--compose-aware needs --init-from).")
                n_skipped += 1
                continue
            cartridge = QwenCartridge.load(init_path, device=device)
            for prm in cartridge.parameters():
                prm.requires_grad_(True)
            cartridge.train()
            distractor_pool = [c for s, c in distractor_bank.items() if s != pid]
            if args.distractor_pool_size and len(distractor_pool) > args.distractor_pool_size:
                distractor_pool = random.sample(distractor_pool, args.distractor_pool_size)
            print(f"  Composition-aware: init from {init_path.parent.name}, "
                  f"{len(distractor_pool)} candidate distractors")
        else:
            cartridge = QwenCartridge.init_from_passage(
                model, tokenizer, context, args.cartridge_len, device
            )

        # Step 2: SELF-STUDY context distillation on the passage's conversations.
        # steps_per_epoch is at least one pass over all conversations.
        steps = max(args.steps_per_epoch, len(conversations))
        print(f"  Training: {args.epochs} epochs × {steps} steps  lr={args.lr}")

        # Save intermediate-epoch checkpoints into sibling dirs so each loads
        # unchanged via evaluate_squad.py --cartridge-dir <dir>_ep<N>.
        checkpoint_fn = None
        if checkpoint_epochs:
            def checkpoint_fn(n, cart, _pid=pid):
                cart.save(Path(f"{args.cartridge_dir}_ep{n}") / _pid / f"cartridge_{args.cartridge_len}.pt")

        # For large corpora, prefill the teacher's corpus-block KV once and reuse it.
        teacher_snapshot, corpus_const_len = None, 0
        if args.cache_teacher:
            teacher_snapshot, corpus_const_len = build_teacher_corpus_cache(
                model, tokenizer, context, device
            )
            print(f"  Cached teacher corpus block: {corpus_const_len} positions")

        QwenCartridge.train_distillation(
            model=model,
            tokenizer=tokenizer,
            cartridge=cartridge,
            conversations=conversations,
            passage_text=context,
            epochs=args.epochs,
            steps_per_epoch=steps,
            lr=args.lr,
            device=device,
            checkpoint_epochs=checkpoint_epochs,
            checkpoint_fn=checkpoint_fn,
            freeze_sink=not args.no_freeze_sink,
            full_conversation_kl=args.full_conversation_kl,
            teacher_corpus_snapshot=teacher_snapshot,
            corpus_const_len=corpus_const_len,
            no_reuse=args.no_reuse,
            distractor_pool=distractor_pool,
            max_distractors=args.max_distractors,
            compose_geom=args.compose_geom,
            compose_shared_sink=args.compose_shared_sink,
        )

        # Step 3: save
        cartridge.eval()
        cartridge.save(out_path)
        n_trained += 1

    print(f"\n{'='*70}")
    print(f"Done.  Trained: {n_trained}  Skipped: {n_skipped}")
    print(f"Cartridges saved in: {cart_dir}/")
    print(f"\nNext: python scripts/evaluate_squad.py --passages-file {args.passages_file}")


if __name__ == "__main__":
    main()
