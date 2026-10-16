"""
benchmark_cost.py — Gate 1 per-query cost benchmark (cartridge vs passage-in-context).

Quality parity (cartridge ≈ teacher) is necessary but not sufficient: Gate 1 asks whether a
trained cartridge beats simply putting the relevant text in context on the COST/quality frontier.
This script measures the per-query inference cost of four conditions on the same SQuAD setup the
quality eval uses, reusing the exact cartridge/RoPE mechanics from generate_answer.

Conditions
----------
  no_context     — question only, no context (reference floor)
  cartridge      — trained cartridge KV preloaded once per passage (load is NOT per-query)
  teacher_cold   — full passage in the prompt, re-prefilled every query (naive RAG)
  teacher_cached — passage KV prefilled ONCE per passage, reused across its queries (the strong
                   baseline in a repeated-query regime; what you'd obviously do for a hot passage)

Per-query metrics (median + p90 over --n-queries, after --warmup): input_tokens, gen_tokens,
prefill_ms, gen_ms, total_ms, kv_positions, peak GPU MB. Also reports one-time costs (cartridge
.pt size; cached-prefix one-time prefill) and an amortization break-even N if --train-seconds is
given (measure it by timing one train_squad_cartridges.py run for a single --passage-ids).

Usage
-----
python scripts/benchmark_cost.py \
    --passages-file data/squad/pilot/passages.jsonl \
    --cartridge-dir cartridges/squad --cartridge-len 256 \
    --n-queries 40 --warmup 5 --scale-name passage
"""
from __future__ import annotations

import json
import time
import argparse
import statistics
import sys
from pathlib import Path

import torch
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).parent))
from cartridge_qwen import (
    get_device, load_model, _build_prefix, CartridgeCache, QwenCartridge, _layer_kv,
)
from evaluate_squad import load_passages

CONDITIONS = ["no_context", "cartridge", "teacher_cold", "teacher_cached"]
METRICS = ["input_tokens", "gen_tokens", "prefill_ms", "gen_ms", "total_ms",
           "kv_positions", "peak_mb"]


def _sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def _stop_token_ids(tokenizer) -> set[int]:
    stop = {tokenizer.eos_token_id}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        stop.add(im_end)
    stop.update(tokenizer.encode("\n", add_special_tokens=False))
    return stop


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _prefilled_cache_from_snapshot(snapshot) -> DynamicCache:
    """Build a fresh DynamicCache seeded with previously-computed prefix K,V per layer.

    The snapshot tensors are never mutated: model attention appends via torch.cat (out-of-place),
    so the same snapshot is safely reused across queries — i.e. the prefix KV is computed once."""
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(snapshot):
        cache.update(k, v, layer_idx)
    return cache


@torch.no_grad()
def run_query(model, query_ids, cache, base, max_new_tokens, device, stop_ids):
    """Greedily decode from `cache` (which already holds `base` positions), prefilling `query_ids`.

    Mirrors generate_answer's loop but splits timing into prefill (first forward over query_ids)
    vs generation (the per-token loop). Returns a metrics dict for this single query."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    cur = torch.tensor([query_ids], dtype=torch.long, device=device)
    n_done = 0
    generated = 0

    _sync(device)
    t0 = time.perf_counter()

    # Prefill: one forward over the whole query suffix/prefix.
    pos = torch.arange(base, base + cur.shape[1], dtype=torch.long, device=device).unsqueeze(0)
    out = model(input_ids=cur, past_key_values=cache, position_ids=pos, use_cache=True, logits_to_keep=1)
    n_done += cur.shape[1]
    _sync(device)
    t_prefill = time.perf_counter()

    nxt = int(out.logits[0, -1].argmax())
    for _ in range(max_new_tokens):
        if nxt in stop_ids:
            break
        generated += 1
        cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
        pos = torch.arange(base + n_done, base + n_done + 1, dtype=torch.long, device=device).unsqueeze(0)
        out = model(input_ids=cur, past_key_values=cache, position_ids=pos, use_cache=True, logits_to_keep=1)
        n_done += 1
        nxt = int(out.logits[0, -1].argmax())

    _sync(device)
    t_end = time.perf_counter()

    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else float("nan")
    return {
        "input_tokens": len(query_ids),
        "gen_tokens": generated,
        "prefill_ms": (t_prefill - t0) * 1e3,
        "gen_ms": (t_end - t_prefill) * 1e3,
        "total_ms": (t_end - t0) * 1e3,
        "kv_positions": base + n_done,
        "peak_mb": peak_mb,
    }


def main():
    ap = argparse.ArgumentParser(description="Gate 1 per-query cost benchmark")
    ap.add_argument("--passages-file", default="data/squad/pilot/passages.jsonl")
    ap.add_argument("--cartridge-dir", default="cartridges/squad")
    ap.add_argument("--cartridge-len", type=int, default=256)
    ap.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--gen-max-tokens", type=int, default=48)
    ap.add_argument("--n-queries", type=int, default=40, help="Queries to time (after warmup)")
    ap.add_argument("--warmup", type=int, default=5, help="Untimed warmup queries")
    ap.add_argument("--results-dir", default="results/squad")
    ap.add_argument("--scale-name", default="passage", help="Label for the output file / report")
    ap.add_argument("--train-seconds", type=float, default=None,
                    help="Measured one-time training wall-clock per cartridge (for amortization)")
    ap.add_argument("--synth-question",
                    default="What is the main subject of this passage and where were they born?",
                    help="Probe question used when a passage record has empty qas (e.g. the 2Wiki "
                         "articles.jsonl, whose questions live in multihop.jsonl). Cost is length-driven, "
                         "so a realistic-length placeholder query gives faithful per-hop cost numbers.")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    model, tokenizer = load_model(args.model_id, device)
    stop_ids = _stop_token_ids(tokenizer)
    num_layers = model.config.num_hidden_layers

    passages = load_passages(Path(args.passages_file))
    cart_dir = Path(args.cartridge_dir)
    need = args.warmup + args.n_queries

    # The constant cacheable prefix for teacher_cached: everything up to and incl "\n\nQuestion: ".
    MARK = "\n\nQuestion: "

    raw = {c: [] for c in CONDITIONS}   # per-query metrics, in execution order
    cart_sizes_mb: list[float] = []
    cached_prefill_ms: list[float] = []

    collected = 0
    for entry in passages:
        if collected >= need:
            break
        pid, title, context, qas = entry["passage_id"], entry["title"], entry["context"], entry["qas"]
        if not qas:                              # 2Wiki articles.jsonl: questions live in multihop.jsonl
            qas = [{"question": args.synth_question}]
        cart_path = cart_dir / pid / f"cartridge_{args.cartridge_len}.pt"
        if not cart_path.exists():
            continue
        correct_cart = QwenCartridge.load(cart_path, device=device)
        cart_len = correct_cart.cartridge_len
        cart_sizes_mb.append(cart_path.stat().st_size / 1e6)

        # One-time per passage: prefill the constant teacher prefix and snapshot its KV.
        sample_full = _build_prefix(qas[0]["question"], context, tokenizer)
        const_str = sample_full[: sample_full.rindex(MARK) + len(MARK)]
        const_ids = tokenizer.encode(const_str, add_special_tokens=False)
        _sync(device)
        _t0 = time.perf_counter()
        with torch.no_grad():
            const_out = model(input_ids=torch.tensor([const_ids], device=device), use_cache=True, logits_to_keep=1)
        _sync(device)
        cached_prefill_ms.append((time.perf_counter() - _t0) * 1e3)
        snapshot = []
        for l in range(num_layers):
            k, v = _layer_kv(const_out.past_key_values, l)   # version-robust DynamicCache read
            snapshot.append((k.clone(), v.clone()))

        for qa in qas:
            if collected >= need:
                break
            q = qa["question"]

            # Token sequences per condition.
            q_only_ids = tokenizer.encode(_build_prefix(q, None, tokenizer), add_special_tokens=False)
            teacher_full = _build_prefix(q, context, tokenizer)
            teacher_ids = tokenizer.encode(teacher_full, add_special_tokens=False)
            cache_len = _common_prefix_len(const_ids, teacher_ids)  # back off if BPE merged the boundary
            suffix_ids = teacher_ids[cache_len:]

            # no_context
            raw["no_context"].append(run_query(
                model, q_only_ids, DynamicCache(), 0, args.gen_max_tokens, device, stop_ids))
            # cartridge (question-only prompt; cartridge supplies the passage)
            raw["cartridge"].append(run_query(
                model, q_only_ids, CartridgeCache(correct_cart, device=device, dtype=model.dtype),
                cart_len, args.gen_max_tokens, device, stop_ids))
            # teacher_cold (full passage re-prefilled)
            raw["teacher_cold"].append(run_query(
                model, teacher_ids, DynamicCache(), 0, args.gen_max_tokens, device, stop_ids))
            # teacher_cached (reuse the one-time passage prefix; only the suffix is per-query)
            raw["teacher_cached"].append(run_query(
                model, suffix_ids, _prefilled_cache_from_snapshot(snapshot),
                cache_len, args.gen_max_tokens, device, stop_ids))

            collected += 1

    if collected == 0:
        print("No cartridges found — train them first.")
        return

    # Drop warmup, aggregate.
    def agg(vals):
        return {
            "median": statistics.median(vals) if vals else float("nan"),
            "p90": (statistics.quantiles(vals, n=10)[8] if len(vals) >= 10 else max(vals)) if vals else float("nan"),
        }

    summary = {}
    for c in CONDITIONS:
        timed = raw[c][args.warmup:]
        summary[c] = {m: agg([r[m] for r in timed]) for m in METRICS}
        summary[c]["n"] = len(timed)

    # ── Report ──────────────────────────────────────────────────────────────────────
    print(f"\n{'='*92}")
    print(f"  GATE 1 COST BENCHMARK — scale='{args.scale_name}'  cartridge_len={args.cartridge_len}  "
          f"(n={summary[CONDITIONS[0]]['n']} timed, {args.warmup} warmup)")
    print(f"{'='*92}")
    hdr = f"  {'condition':<16}{'in_tok':>8}{'gen_tok':>8}{'prefill_ms':>12}{'gen_ms':>10}{'total_ms':>10}{'kv_pos':>8}{'peak_MB':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c in CONDITIONS:
        s = summary[c]
        print(f"  {c:<16}{s['input_tokens']['median']:>8.0f}{s['gen_tokens']['median']:>8.0f}"
              f"{s['prefill_ms']['median']:>12.1f}{s['gen_ms']['median']:>10.1f}{s['total_ms']['median']:>10.1f}"
              f"{s['kv_positions']['median']:>8.0f}{s['peak_mb']['median']:>10.1f}")
    print("  (medians; p90 in JSON)")

    # One-time costs.
    avg_cart_mb = statistics.mean(cart_sizes_mb) if cart_sizes_mb else float("nan")
    avg_cached_prefill = statistics.mean(cached_prefill_ms) if cached_prefill_ms else float("nan")
    print(f"\n  One-time: cartridge .pt = {avg_cart_mb:.1f} MB/passage   "
          f"teacher_cached prefix prefill = {avg_cached_prefill:.1f} ms/passage (paid once)")
    if args.train_seconds is not None:
        print(f"            cartridge training = {args.train_seconds:.1f} s/passage (measured)")

    # Amortization break-even N (on per-query latency).
    cart_ms = summary["cartridge"]["total_ms"]["median"]
    cold_ms = summary["teacher_cold"]["total_ms"]["median"]
    cached_ms = summary["teacher_cached"]["total_ms"]["median"]

    def breakeven(train_s, per_query_cart, per_query_base):
        gain = per_query_base - per_query_cart  # ms saved per query by using the cartridge
        if gain <= 0:
            return None  # cartridge is not faster per query → never amortizes
        return (train_s * 1e3) / gain

    print(f"\n  Per-query latency: cartridge={cart_ms:.1f} ms  teacher_cold={cold_ms:.1f} ms  "
          f"teacher_cached={cached_ms:.1f} ms")
    if args.train_seconds is not None:
        for label, base_ms in [("teacher_cold", cold_ms), ("teacher_cached", cached_ms)]:
            n = breakeven(args.train_seconds, cart_ms, base_ms)
            if n is None:
                print(f"    vs {label:<14}: cartridge is NOT faster per query → never amortizes (∞)")
            else:
                print(f"    vs {label:<14}: break-even after ~{n:,.0f} queries to the same passage")
    else:
        print("    (pass --train-seconds <s> to compute amortization break-even N)")

    # Persist.
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cost_{args.scale_name}.json"
    out_path.write_text(json.dumps({
        "config": vars(args),
        "summary": summary,
        "one_time": {
            "cartridge_mb_per_passage": avg_cart_mb,
            "teacher_cached_prefix_prefill_ms": avg_cached_prefill,
            "train_seconds_per_passage": args.train_seconds,
        },
        "per_query_total_ms": {"cartridge": cart_ms, "teacher_cold": cold_ms, "teacher_cached": cached_ms},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
