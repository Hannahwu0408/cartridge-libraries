"""
train_joint.py — CAS-style JOINT mixed-visibility training of a cartridge pool.

Replicates the fix from "Cartridges at Scale" (arXiv 2606.04557): instead of training each
cartridge in isolation (or with FROZEN distractors, as our compose-aware did), train the WHOLE
pool together. Every cartridge is a trainable parameter under ONE optimizer; each step samples a
document's self-study conversation and, with prob (1 - P_iso), prepends k~U(1,M) RANDOM distractor
cartridges (plain KV concatenation). Gradients flow through ALL active cartridges, so each learns
to be selectable when relevant AND non-interfering when it's someone else's distractor. This is
what teaches the frozen model to selectively attend across independently-initialized KV prefixes
(CAS: 20 isolated carts compose at 26.0 → jointly-trained 77.8, oracle 79.0).

At synth scale (~40 len-64 cartridges ≈ 0.8 GB) the whole pool fits on GPU, so no budget manager /
rotation is needed. The teacher stays clean (the relevant document's article in context); the
composition mechanism is `concat` (StackedCartridgeCache), matching CAS.

Usage
-----
python scripts/train_joint.py \
    --passages-file data/synth/articles_mm.jsonl \
    --conversations-file data/synth/selfstudy_ca_all.jsonl \
    --model-id Qwen/Qwen3-4B --cartridge-len 64 \
    --init-from cartridges/synth_mm --cartridge-dir cartridges/synth_joint \
    --epochs 6 --lr 3e-4 --p-iso 0.75 --max-distractors 5
"""
from __future__ import annotations

import json
import random
import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cartridge_qwen import get_device, load_model, QwenCartridge  # noqa: E402


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="Joint mixed-visibility cartridge training (CAS-style)")
    ap.add_argument("--passages-file", required=True)
    ap.add_argument("--conversations-file", required=True)
    ap.add_argument("--bridge-conversations", default=None,
                    help="S2 co-training: two-gold bridge convs (generate_bridge_data.py). A fraction "
                         "--p-bridge of steps composes BOTH gold cartridges (both TRAINABLE) against a "
                         "two-passage teacher → teaches co-attention in the KV (in-attention synthesis)")
    ap.add_argument("--p-bridge", type=float, default=0.25,
                    help="fraction of steps that are two-gold bridge examples (S2)")
    ap.add_argument("--descriptive-conversations", default=None,
                    help="Phase 12-H ADDRESSABILITY: descriptive/relational self-study (refers to the "
                         "entity by its in-article descriptors). Mixed into single-conv steps at "
                         "--p-descriptive so the cartridge becomes queryable by description.")
    ap.add_argument("--p-descriptive", type=float, default=0.0,
                    help="fraction of SINGLE-conv steps drawn from the descriptive self-study "
                         "(0.5 = 50/50 base:descriptive; 0.6 = 40/60)")
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--cartridge-len", type=int, default=64)
    ap.add_argument("--cartridge-dir", required=True, help="output dir for the trained pool")
    ap.add_argument("--init-from", default=None,
                    help="dir of existing cartridges to init from (else init_from_passage)")
    ap.add_argument("--passage-ids", default=None, help="comma-sep subset (default: all)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=200,
                    help="linear LR warmup (CAS-style); keeps early updates stable")
    ap.add_argument("--final-lr-frac", type=float, default=0.02,
                    help="final LR as a fraction of peak (CAS slow linear decay = 0.02)")
    ap.add_argument("--p-iso", type=float, default=0.75,
                    help="prob a step shows ONLY the relevant cartridge (else add distractors)")
    ap.add_argument("--max-distractors", type=int, default=5, help="M in k~U(1,M)")
    ap.add_argument("--compose-geom", choices=["concat", "gnorm"], default="concat",
                    help="composition geometry to train under (gnorm → a gate-compatible pool)")
    ap.add_argument("--max-convs-per-cart", type=int, default=None,
                    help="subsample self-study convs per cartridge (2Wiki big self-study has ~3k/article; "
                         "cap for a tractable epoch — at low compression the facts fit in far fewer)")
    ap.add_argument("--max-answer-tokens", type=int, default=48,
                    help="cap the supervised answer window (bounds the retained graph → memory)")
    ap.add_argument("--max-question-tokens", type=int, default=96,
                    help="drop convs whose question exceeds N tokens (malformed long-question convs "
                         "spike memory at len-256); 0 = keep all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adam8bit", action="store_true",
                    help="use bitsandbytes Adam8bit (optimizer states ~4x smaller; fits large cart pools "
                         "like the 196-cart 2Wiki_v2 on one 24GB GPU with bridge pairs co-resident)")
    ap.add_argument("--distractor-mode", choices=["uniform", "local"], default="uniform",
                    help="CL-1: 'local' samples distractors from the target's kNN neighbors (hard "
                         "negatives = same-family confusables) instead of uniformly; needs --topics-file")
    ap.add_argument("--topics-file", default=None,
                    help="topics.json from build_topic_layer.py (provides knn_neighbors, sim-weighted)")
    ap.add_argument("--local-p", type=float, default=0.8,
                    help="local mode: prob each distractor draw comes from the kNN pool (else uniform)")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Device: {device}")
    model, tokenizer = load_model(args.model_id, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    passages = {p["passage_id"]: p["context"] for p in load_jsonl(args.passages_file)}
    if args.passage_ids:
        wanted = {s.strip() for s in args.passage_ids.split(",")}
        passages = {k: v for k, v in passages.items() if k in wanted}
    convs_by_pid: dict[str, list[dict]] = {}
    for rec in load_jsonl(args.conversations_file):
        if rec["passage_id"] in passages:
            convs_by_pid.setdefault(rec["passage_id"], []).append(rec)
    if args.max_question_tokens and args.max_question_tokens > 0:
        def q_ok(rec):
            return len(tokenizer.encode(rec.get("question", ""), add_special_tokens=False)) \
                <= args.max_question_tokens
        n0 = sum(len(v) for v in convs_by_pid.values())
        convs_by_pid = {p: [c for c in cs if q_ok(c)] for p, cs in convs_by_pid.items()}
        print(f"Dropped {n0 - sum(len(v) for v in convs_by_pid.values())}/{n0} long-question convs "
              f"(>{args.max_question_tokens} tok)")
    if args.max_convs_per_cart:
        convs_by_pid = {p: (random.sample(cs, args.max_convs_per_cart)
                            if len(cs) > args.max_convs_per_cart else cs)
                        for p, cs in convs_by_pid.items()}
    pids = [p for p in passages if convs_by_pid.get(p)]
    print(f"Pool: {len(pids)} cartridges, "
          f"{sum(len(convs_by_pid[p]) for p in pids)} conversations")

    # CL-1: kNN hard-negative distractors. knn[pid] = [(neighbor_pid, cos_sim), ...] from topics.json —
    # kNN-adjacent carts are the same-family confusables the cart is actually co-retrieved with.
    knn: dict[str, list[tuple[str, float]]] | None = None
    if args.distractor_mode == "local":
        if not args.topics_file:
            ap.error("--distractor-mode local requires --topics-file (from build_topic_layer.py)")
        pset = set(pids)
        raw = json.loads(Path(args.topics_file).read_text())["knn_neighbors"]
        knn = {p: [(n, float(s)) for n, s in raw.get(p, []) if n in pset] for p in pids}
        print(f"local distractors: kNN lists for {sum(1 for p in pids if knn[p])}/{len(pids)} carts "
              f"(local-p={args.local_p}, sim-weighted)")

    def pick_distractors(anchors: set[str], k: int) -> list[str]:
        """k distractor pids. uniform: random from pool. local: each draw w.p. --local-p from the
        anchors' merged kNN pool (sim-weighted, no replacement), else uniform from the rest."""
        rest = [p for p in pids if p not in anchors]
        if knn is None:
            return random.sample(rest, k)
        near: dict[str, float] = {}
        for a in anchors:
            for n, s in knn.get(a, []):
                if n not in anchors:
                    near[n] = max(near.get(n, 0.0), s)
        picked: list[str] = []
        for _ in range(k):
            cand = [(n, s) for n, s in near.items() if n not in picked]
            if cand and random.random() < args.local_p:
                names, sims = zip(*cand)
                picked.append(random.choices(names, weights=sims, k=1)[0])
            else:
                pool = [p for p in rest if p not in picked]
                if not pool:
                    break
                picked.append(random.choice(pool))
        return picked

    # S2: two-gold bridge convs whose BOTH golds are in the pool (both get gradient → co-attention).
    bridge_convs = []
    if args.bridge_conversations:
        if args.compose_geom != "gnorm":
            ap.error("--bridge-conversations requires --compose-geom gnorm (the two-gold path is gnorm-only)")
        pidset = set(pids)
        for rec in load_jsonl(args.bridge_conversations):
            ab = rec.get("passage_ids", [None, None])[:2]
            if len(ab) == 2 and ab[0] in pidset and ab[1] in pidset:
                if args.max_question_tokens and len(tokenizer.encode(
                        rec.get("question", ""), add_special_tokens=False)) > args.max_question_tokens:
                    continue
                bridge_convs.append(rec)
        print(f"S2 bridge convs: {len(bridge_convs)} (p_bridge={args.p_bridge})")
    n_single = sum(len(convs_by_pid[p]) for p in pids)
    n_bridge_per_epoch = (round(args.p_bridge / (1 - args.p_bridge) * n_single)
                          if bridge_convs and args.p_bridge < 1 else 0)

    # 12-H addressability: descriptive/relational single-cartridge convs, mixed at --p-descriptive.
    desc_convs = []
    if args.descriptive_conversations:
        pset = set(pids)
        for rec in load_jsonl(args.descriptive_conversations):
            if rec.get("passage_id") in pset:
                if args.max_question_tokens and len(tokenizer.encode(
                        rec.get("question", ""), add_special_tokens=False)) > args.max_question_tokens:
                    continue
                desc_convs.append(rec)
        print(f"Descriptive convs: {len(desc_convs)} (p_descriptive={args.p_descriptive})")
    n_desc_per_epoch = (round(args.p_descriptive / (1 - args.p_descriptive) * n_single)
                        if desc_convs and args.p_descriptive < 1 else 0)

    # Load/init the whole pool as TRAINABLE (one Adam over all of them).
    init_dir = Path(args.init_from) if args.init_from else None
    carts: dict[str, QwenCartridge] = {}
    for pid in pids:
        if init_dir is not None and (init_dir / pid / f"cartridge_{args.cartridge_len}.pt").exists():
            c = QwenCartridge.load(init_dir / pid / f"cartridge_{args.cartridge_len}.pt", device=device)
        else:
            c = QwenCartridge.init_from_passage(model, tokenizer, passages[pid],
                                                args.cartridge_len, device)
        c.train()
        for p in c.parameters():
            p.requires_grad_(True)
        carts[pid] = c
    all_params = [p for c in carts.values() for p in c.parameters()]
    if args.adam8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(all_params, lr=args.lr)
        print("optimizer: bitsandbytes Adam8bit")
    else:
        opt = torch.optim.Adam(all_params, lr=args.lr)
    # CAS-style schedule: short linear warmup, then SLOW linear decay to final_lr_frac·peak,
    # stepped PER optimizer step (not per epoch) so the effective LR stays high throughout.
    steps_per_epoch = n_single + n_bridge_per_epoch + n_desc_per_epoch
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup = min(args.warmup_steps, total_steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 1.0 - (1.0 - args.final_lr_frac) * min(prog, 1.0)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print(f"Trainable params: {sum(p.numel() for p in all_params)/1e6:.1f}M across {len(carts)} carts"
          f"  (total_steps~{total_steps}, warmup {warmup})")

    cart_len = args.cartridge_len
    cuda = device == "cuda"
    for epoch in range(args.epochs):
        steps = [("single", pid, conv) for pid in pids for conv in convs_by_pid[pid]]
        # S2: mix in two-gold bridge steps to reach a p_bridge fraction of the epoch.
        steps += [("bridge", None, random.choice(bridge_convs)) for _ in range(n_bridge_per_epoch)]
        # 12-H: mix in descriptive single-cartridge steps to reach a p_descriptive fraction.
        if n_desc_per_epoch:
            steps += [("single", d["passage_id"], d)
                      for d in random.choices(desc_convs, k=n_desc_per_epoch)]
        random.shuffle(steps)
        total, n, skipped, oom, nb = 0.0, 0, 0, 0, 0
        for kind, pid, conv in steps:
            try:
                if kind == "single":
                    X = carts[pid]
                    distractors, x_pos = None, 0
                    if len(pids) > 1 and random.random() > args.p_iso:
                        k = random.randint(1, min(args.max_distractors, len(pids) - 1))
                        others = pick_distractors({pid}, k)
                        distractors = [carts[o] for o in others]
                        x_pos = random.randint(0, len(distractors))   # X's block among distractors
                    loss = QwenCartridge._distill_loss_for_conv(
                        model, tokenizer, X, conv, passages[pid], cart_len, device,
                        full_conversation_kl=False, max_answer_tokens=args.max_answer_tokens,
                        distractors=distractors, x_pos=x_pos, compose_geom=args.compose_geom,
                    )
                else:  # bridge: compose BOTH gold cartridges (both TRAINABLE) vs a two-passage teacher
                    a, b = conv["passage_ids"][:2]
                    distractors = None
                    if len(pids) > 2 and random.random() > args.p_iso:
                        k = random.randint(1, min(args.max_distractors, len(pids) - 2))
                        distractors = [carts[o] for o in pick_distractors({a, b}, k)]
                    combined = f"Passage 1:\n{passages[a].strip()}\n\nPassage 2:\n{passages[b].strip()}"
                    loss = QwenCartridge._distill_loss_for_conv(
                        model, tokenizer, carts[a], conv, combined, cart_len, device,
                        full_conversation_kl=False, max_answer_tokens=args.max_answer_tokens,
                        distractors=distractors, compose_geom="gnorm",
                        gold_cartridges=[carts[a], carts[b]],
                    )
                    nb += 1
                if loss is None:
                    skipped += 1
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)  # only active carts have grad
                opt.step()
                sched.step()                       # per-step LR (warmup + slow linear decay)
                total += loss.item()
                n += 1
                del loss
            except torch.cuda.OutOfMemoryError:
                oom += 1
                opt.zero_grad(set_to_none=True)
                if cuda:
                    torch.cuda.empty_cache()
        print(f"epoch {epoch+1}/{args.epochs}  loss={total/max(n,1):.4f}  "
              f"steps={n} (bridge={nb})  skipped={skipped}  oom={oom}  lr={sched.get_last_lr()[0]:.2e}")

    out_dir = Path(args.cartridge_dir)
    for pid, c in carts.items():
        c.eval()
        c.save(out_dir / pid / f"cartridge_{cart_len}.pt")
    print(f"Saved {len(carts)} jointly-trained cartridges → {out_dir}")


if __name__ == "__main__":
    main()
