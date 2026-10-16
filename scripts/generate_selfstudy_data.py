"""
generate_selfstudy_data.py — SELF-STUDY conversational training data, CAS-style.

Follows the improved self-study synthesis of "Cartridges at Scale" (arXiv 2606.04557, §2.3):
  • BATCHED multi-question generation: one call to the question model produces N diverse
    questions (returned as a JSON array), with an instruction to vary style (factual recall,
    comparison, reasoning, detail) and cover DIFFERENT facts — ~N× fewer calls AND better fact
    coverage than one-at-a-time sampling (which collides on dedup).
  • DECOUPLED question model (optional --question-model): use a larger/stronger model to WRITE
    the questions, while the target model (--model-id) ANSWERS them (its own distribution is the
    distillation target). CAS's biggest single data-quality lever.
  • GREEDY, BATCHED answers: the target model answers questions deterministically (temp 0.0) in
    left-padded batches for throughput. Answers are its own fluent generations → sharp KL targets.
  • Four seed modes (question / structuring / summarization / use_case); "creative" is excluded
    (CAS drops it — open-ended prompts are less suited for distillation).

Output — one record per conversation:
    {"passage_id": ..., "seed_type": ..., "question": ..., "answer": ...}

Usage
-----
python scripts/generate_selfstudy_data.py \
    --passages-file data/2wiki_rt/articles.jsonl --out-file data/2wiki_rt/selfstudy_big.jsonl \
    --model-id Qwen/Qwen3-4B --per-passage 300 --n-per-call 20
#  decoupled stronger question model (needs memory for both):
#    --question-model Qwen/Qwen3-8B
"""
from __future__ import annotations

import re
import json
import random
import argparse
import collections
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from cartridge_qwen import load_model, _build_conv_prefix, _layer_kv

# MUST stay byte-identical to router.py/text_rag.py's SHORT. With --terse-frac, this instruction is
# appended to a fraction of questions BEFORE the answer is generated AND saved into the training
# record, so (a) turn B decodes a span, (b) the distillation teacher sees the exact context that
# produced that answer, and (c) eval-time --short-answer is in-distribution for the cartridge.
# This is the style-lock intervention (§5.4c): cartridges distilled only on full-sentence answers
# confabulate under a terse instruction; terse-inclusive self-study makes the format trained.
SHORT = ("  Answer with ONLY the short answer itself (a few words, e.g. a name, place, or date) — "
         "no explanation, no full sentence.")

# GROUNDING CONSTRAINT for the schema-filling seed mode (--ground-structured).
# Measured on the fidelity30 testbed with the 8B answerer: `structuring` is 20.1% contradicted vs
# 4.8% for plain `question` — and with the 4B answerer it was 29% vs 18%, i.e. the gap survives a
# model upgrade because the FORMAT causes it, not the model. Every sampled judge reason is the same
# failure: the requested table/JSON/YAML has fields the passage never fills, so the model completes
# the schema from parametric knowledge (inventing dates, renaming a school, assigning awards to the
# wrong film). Telling it to leave gaps empty attacks that mechanism directly.
GROUND = ("  Use ONLY information explicitly stated in the passage above. Do not add facts from "
          "outside knowledge, and do not invent dates, titles, names, or categories that are not "
          "written there. If the passage does not state something, omit that field, row, or entry "
          "entirely rather than guessing or filling it in.")

# All three generative modes. Originally scoped to `structuring` alone (the only mode then measured),
# which turned the first widened run into a clean controlled experiment: structuring fell 22.4% →
# 11.3% pooled (p<1e-5, paired on identical questions) while summarization and use_case came back
# BYTE-IDENTICAL — same seed, untouched prompt, so same questions and same greedy answers. That
# isolates the effect to the constraint and rules out collateral drift. Their baselines are now
# measured too (use_case 12.5%, summarization 8.3%, vs 4.8% for plain `question`) and their sampled
# failures are the same mechanism — completing an open-ended request from parametric memory rather
# than the passage ("victim Laura Palmer", awards dated from world knowledge) — so the constraint
# applies for the same reason. `question` stays out on ONE measured ground and one untested worry:
# MEASURED — at 4.8% it is already near the floor, and the mechanism grounding fixes (a schema whose
# fields the passage cannot fill) does not exist for direct QA, so there is little headroom to buy.
# UNTESTED — the claim that constraining direct QA would teach refusal-ish hedging on the deployed
# lane. No arm has ever run it. The worry is not idle (refusals are 52% of the pool196 chain's residual
# failures) but it is a hypothesis; --ground-modes exists to settle it. Do not cite it as a finding.
_GROUNDED_MODES = {"structuring", "summarization", "use_case"}


# ── Batched multi-request seed prompts (each asks for N items as a JSON array) ──────────

def _multi_prompt(mode: str, n: int) -> str:
    common = (f'Output ONLY a JSON array of exactly {n} strings, e.g. '
              f'["...", "...", ...]. No other text, no markdown fences, no explanation.')
    if mode == "question":
        return (
            f"Generate {n} diverse questions that test knowledge of the information in the "
            f"passage above. Each question must cover a DIFFERENT fact, detail, or aspect. Vary "
            f"the style: mix factual recall, comparison, reasoning, and detail-oriented questions. "
            f"Include specific details (names, titles, dates, numbers, places) in each question so "
            f"it is clear what is being asked. {common}")
    if mode == "structuring":
        return (
            f"Generate {n} distinct chat messages, each instructing an assistant to extract some "
            f"of the key facts in the passage above into a structured format (JSON, YAML, a bulleted "
            f"list, or a table), each focused on a DIFFERENT set of details. {common}")
    if mode == "summarization":
        return (
            f"Generate {n} distinct chat messages, each asking an assistant to summarize a DIFFERENT "
            f"aspect or section of the passage above. {common}")
    if mode == "use_case":
        return (
            f"Generate {n} distinct chat messages, each asking an assistant to apply a DIFFERENT fact "
            f"from the passage above to a realistic, practical task. {common}")
    if mode == "relational":
        # ADDRESSABILITY (Phase 12-H): refer to the subject by a DESCRIPTOR/ROLE/RELATION from the
        # passage instead of by name, so the cartridge's facts become reachable from descriptive
        # queries ("the director of X born where?") — the framing a 2-hop bridge query actually uses.
        # Crucially, cycle through ALL of the subject's descriptors (multiple films/works/relations),
        # incl. the SAME fact via different descriptors, so the fact is reachable from EACH of them.
        return (
            f"Generate {n} diverse questions about the passage above, but REFER TO ITS MAIN SUBJECT "
            f"BY A DESCRIPTION, ROLE, OR RELATION stated in the passage — NOT by name (e.g. 'the "
            f"director of <film>', 'the author of <work>', 'the founder of <org>', 'the father of "
            f"<person>', 'the person who <did X in the passage>'). If the subject is associated with "
            f"SEVERAL works, roles, or relations in the passage (multiple films, books, positions, "
            f"family members), USE A DIFFERENT ONE in different questions so the subject is reachable "
            f"from EACH of its descriptors — it is GOOD to ask the SAME fact (e.g. birthplace) via "
            f"several different descriptors. Every question must be answerable from the passage and "
            f"must NOT mention the subject's own name. {common}")
    raise ValueError(mode)


_MODES = ["question", "question", "structuring", "summarization", "use_case"]  # weight 'question'


def _parse_items(text: str) -> list[str]:
    """Parse a JSON array of strings from the model output, robust to fences / stray text."""
    s = text.strip()
    start, end = s.find("["), s.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            arr = json.loads(s[start:end + 1])
            items = [str(x).strip() for x in arr if str(x).strip()]
            if items:
                return items
        except json.JSONDecodeError:
            pass
    # fallback: numbered / bulleted lines
    lines = [re.sub(r'^\s*(?:\d+[.)]|[-*])\s*', '', l).strip().strip('"')
             for l in s.splitlines() if l.strip()]
    return [l for l in lines if len(l) > 5 and l not in ("[", "]")]


# ── Generation ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate_one(model, tokenizer, prefix_text: str, max_new_tokens: int,
                  temperature: float, device: str) -> str:
    ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        inp, attention_mask=torch.ones_like(inp), max_new_tokens=max_new_tokens,
        do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
        top_p=0.95 if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, inp.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def _generate_batch(model, tokenizer, prefixes: list[str], max_new_tokens: int,
                    temperature: float, device: str) -> list[str]:
    """Batched (left-padded) generation → one completion per prefix. temp=0 → greedy."""
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prefixes, return_tensors="pt", padding=True,
                    add_special_tokens=False).to(device)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens,
        do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
        top_p=0.95 if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    tokenizer.padding_side = prev_side
    plen = enc["input_ids"].shape[1]
    return [tokenizer.decode(out[i, plen:], skip_special_tokens=True).strip()
            for i in range(out.shape[0])]


# ── shared-passage prefix cache ──────────────────────────────────────────────────────────
# Every answer prompt is [system: _CONV_SYSTEM(passage)][user: question], so the ~234-token article is
# re-prefilled once per conversation — 2,000x per cartridge, ~0.47M redundant prefill tokens. All the
# conversations of one article share that exact token prefix, so it can be computed ONCE and reused.
# Pattern (and the safety argument) taken from benchmark_cost.py: the snapshot is never mutated because
# attention appends via torch.cat out-of-place, so one snapshot serves every batch of the article.

def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# Probes for boundary discovery. Their FIRST user-content tokens must all differ, or the discovered
# "shared prefix" would run into question text: the per-batch guard would then reject almost every real
# question and silently fall back to the uncached path — the risk of a new code path with none of the
# speedup, and no signal that it happened. Four probes with distinct leading characters, plus an
# explicit containment assertion, make the boundary a property of the template rather than of luck.
_PREFIX_PROBES = ["A?", "zebra crossing?", "7 of them?", "Was ist das?"]


@torch.no_grad()
def build_passage_prefix(model, tokenizer, passage: str, device: str, verbose: bool = False):
    """Return (prefix_ids, snapshot) for a passage, or None if no usable shared prefix exists.

    The boundary is DISCOVERED, not assumed: tokenize several prompts differing ONLY in the question
    and take the common prefix across all of them. That covers [system header][passage][system end]
    [user header] without hard-coding chat-template internals, and BPE merges at the question boundary
    resolve safely — a token that fuses the header's last character with the question's first differs
    across probes, so the prefix simply ends before it.

    Fails safe: an implausibly short prefix, or one that has swallowed probe text, returns None and the
    caller uses the uncached path.
    """
    encs = [tokenizer(_build_conv_prefix(q, passage, tokenizer),
                      add_special_tokens=False)["input_ids"] for q in _PREFIX_PROBES]
    p = min(_common_prefix_len(encs[0], e) for e in encs[1:])
    if p < 16 or p >= min(len(e) for e in encs):
        return None
    text = tokenizer.decode(encs[0][:p])
    if any(q[:6] in text for q in _PREFIX_PROBES):      # boundary ran into question text
        return None
    if verbose:
        print(f"  prefix-cache: boundary at {p} tokens, ends with {text[-60:]!r}")
    ids = torch.tensor([encs[0][:p]], dtype=torch.long, device=device)
    out = model(input_ids=ids, use_cache=True)
    # _layer_kv, not raw indexing: DynamicCache has had three different layouts across transformers
    # versions (`cache.layers[l].keys` / `cache.key_cache[l]` / subscriptable), and the training server runs 5.12.1.
    snap = [_layer_kv(out.past_key_values, l) for l in range(model.config.num_hidden_layers)]
    return encs[0][:p], snap


@torch.no_grad()
def _generate_batch_cached(model, tokenizer, prefixes: list[str], max_new_tokens: int,
                           temperature: float, device: str, prefix_ids, snapshot) -> list[str]:
    """Same contract as _generate_batch, but the shared passage prefix comes from `snapshot`.

    Only the per-question SUFFIX is prefilled. Suffixes are LEFT-padded so every sequence ends at the
    same position (generation stays aligned); the attention mask is ones over the cached prefix
    concatenated with the suffix mask, and HF derives position_ids from that mask's cumsum — which is
    precisely what makes left padding correct here.
    """
    from transformers import DynamicCache
    P, B = len(prefix_ids), len(prefixes)
    suffixes = []
    for text in prefixes:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids[:P] != prefix_ids:                  # not this article's prefix → caller must fall back
            return None
        suffixes.append(ids[P:])
    S = max(len(s) for s in suffixes)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    inp = torch.full((B, S), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((B, P + S), dtype=torch.long, device=device)
    mask[:, :P] = 1
    for i, s in enumerate(suffixes):               # left-pad the suffix
        inp[i, S - len(s):] = torch.tensor(s, dtype=torch.long, device=device)
        mask[i, P + S - len(s):] = 1

    cache = DynamicCache()
    for l, (k, v) in enumerate(snapshot):
        cache.update(k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1), l)

    out = model.generate(
        input_ids=inp, attention_mask=mask, past_key_values=cache,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
        top_p=0.95 if temperature > 0 else None,
        pad_token_id=pad_id,
    )
    return [tokenizer.decode(out[i, S:], skip_special_tokens=True).strip() for i in range(B)]


def gen_answers(model, tokenizer, prefixes, max_new_tokens, temperature, device, cached=None):
    """Dispatcher: try the cached path when available, fall back to the reference path on any refusal.

    The uncached _generate_batch is the reference implementation and is never modified — every result
    in this project came from it, so it stays byte-reachable for the A/B validation."""
    if cached is not None:
        got = _generate_batch_cached(model, tokenizer, prefixes, max_new_tokens,
                                     temperature, device, cached[0], cached[1])
        if got is not None:
            return got
        gen_answers.fallbacks = getattr(gen_answers, "fallbacks", 0) + 1
    return _generate_batch(model, tokenizer, prefixes, max_new_tokens, temperature, device)


def load_passages(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _reanswer(args, passages, a_model, tokenizer, device):
    """Re-answer an existing self-study file's questions with a different --model-id.

    The M_A axis of the generator bakeoff on a single GPU: the source file supplies M_Q's questions
    (already generated, on disk), so the question model never has to be co-resident with the answer
    model. Questions, seed_type and passage_id are copied verbatim — only `answer` is regenerated —
    which additionally removes question-sampling noise from the M_A comparison.
    """
    by_pid = {p["passage_id"]: p["context"] for p in passages}
    src = [json.loads(l) for l in Path(args.reanswer).read_text(encoding="utf-8").splitlines()
           if l.strip()]
    src = [c for c in src if c["passage_id"] in by_pid]
    if not src:
        sys.exit(f"ABORT: no conversations in {args.reanswer} match the selected passages.")

    grouped: dict[str, list[dict]] = {}
    for c in src:
        grouped.setdefault(c["passage_id"], []).append(c)
    print(f"Re-answering {len(src)} questions over {len(grouped)} passages "
          f"from {args.reanswer} with {args.model_id}")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written, n_empty = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (pid, convs) in enumerate(grouped.items()):
            passage = by_pid[pid]
            cached = (build_passage_prefix(a_model, tokenizer, passage, device)
                      if args.prefix_cache else None)
            for b in range(0, len(convs), args.answer_batch_size):
                chunk = convs[b:b + args.answer_batch_size]
                # --terse-frac, same semantics as the main path (:600): SHORT is appended BEFORE
                # answering so turn B decodes a span, AND kept in the stored question so eval-time
                # --short-answer is in-distribution. Contrast GROUND below, which shapes the answer
                # but is stripped from the record because the cartridge never sees it at inference.
                terse = [args.terse_frac > 0 and random.random() < args.terse_frac for _ in chunk]
                qs = [c["question"] + (SHORT if t else "") for c, t in zip(chunk, terse)]
                # same asked-vs-stored split as the main path: grounding shapes the answer, the
                # stored question stays as it was in the source file (plus SHORT when terse)
                prefixes = [_build_conv_prefix(
                    q + (GROUND if (args.ground_structured
                                    and c.get("seed_type") in _GROUNDED_MODES) else ""),
                    passage, tokenizer) for c, q in zip(chunk, qs)]
                answers = gen_answers(a_model, tokenizer, prefixes, args.a_max_tokens,
                                      args.answer_temperature, device, cached)
                for c, q, ans in zip(chunk, qs, answers):
                    if not ans:            # same drop rule as the main path
                        n_empty += 1
                        continue
                    f.write(json.dumps({"passage_id": c["passage_id"],
                                        "seed_type": c.get("seed_type", "?"),
                                        "question": q, "answer": ans},
                                       ensure_ascii=False) + "\n")
                    n_written += 1
            f.flush()
            print(f"[{i+1}/{len(grouped)}] {pid[:40]:<40} {len(convs)} questions re-answered")

    print(f"\n{'='*60}\nWrote {n_written} conversations → {out_path}"
          f"{f' ({n_empty} dropped: empty answer)' if n_empty else ''}")


def main():
    ap = argparse.ArgumentParser(description="CAS-style self-study data generation")
    ap.add_argument("--passages-file", default="data/squad/pilot/passages.jsonl")
    ap.add_argument("--out-file", default="data/squad/pilot/selfstudy.jsonl")
    ap.add_argument("--per-passage", type=int, default=300,
                    help="target distinct conversations per passage")
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B", help="ANSWER model (the target)")
    ap.add_argument("--question-model", default=None,
                    help="optional stronger model to WRITE questions (decoupled M_Q); "
                         "default = same as --model-id. Loading two models needs the memory.")
    ap.add_argument("--n-per-call", type=int, default=20, help="questions per generation call")
    ap.add_argument("--q-max-tokens", type=int, default=640, help="tokens for the N-question call")
    ap.add_argument("--a-max-tokens", type=int, default=160)
    ap.add_argument("--q-temperature", type=float, default=0.6, help="question sampling temp")
    ap.add_argument("--answer-temperature", type=float, default=0.0,
                    help="answer temp (0 = greedy, deterministic distillation targets)")
    ap.add_argument("--answer-batch-size", type=int, default=8)
    ap.add_argument("--question-batch-rounds", type=int, default=1, metavar="R",
                    help="Run the next R question-generation rounds as ONE batched call. Question "
                         "generation is ~87%% of pipeline cost at 2000/pp (248 sequential rounds x 640 "
                         "tokens per article) and was the last unbatched stage. R also lifts the "
                         "effective ANSWER batch: --n-per-call 20 caps a single round at 20 questions, "
                         "so --answer-batch-size 32 is unreachable at R=1. Tradeoff: the R rounds cannot "
                         "see each other's questions, so within-batch duplicates are filtered after "
                         "generation instead of prevented (yield loss, reported as dup-waste). "
                         "R=1 (default) is byte-identical to the pre-batching behaviour.")
    ap.add_argument("--prefix-cache", action="store_true",
                    help="Prefill each passage's shared prompt prefix ONCE per article and reuse it for "
                         "every answer in that article, instead of re-prefilling the ~234-token passage "
                         "per conversation (2,000x per cartridge). Falls back to the uncached path "
                         "automatically for any article without a usable shared prefix. NOTE this "
                         "changes batch composition, so greedy outputs can differ in the same way a "
                         "batch-size change already does — validate output agreement against "
                         "the uncached path before using it to produce training data.")
    ap.add_argument("--terse-frac", type=float, default=0.0,
                    help="fraction of questions (per-question, seeded) that get the terse-span "
                         "instruction appended — and KEEP it in the saved record. 0 = current "
                         "behavior (all conversational); 1.0 = pure-terse arm; 0.5 = mixed arm.")
    ap.add_argument("--ground-structured", action="store_true",
                    help="append the GROUND constraint, before answering, to the generative seed "
                         "modes listed in _GROUNDED_MODES (structuring, summarization, use_case) so "
                         "the model omits what the passage does not state instead of completing "
                         "from parametric memory. NOTE this is independent of --modes: --modes "
                         "chooses what to GENERATE, _GROUNDED_MODES chooses what to GROUND. "
                         "Measured: structuring 22.4%% -> 11.3%% (p<1e-5); plain `question` mode is "
                         "deliberately never grounded (already 4.8%%).")
    ap.add_argument("--ground-modes", default=None, metavar="M1,M2",
                    help="Override which seed modes --ground-structured applies to (default: "
                         "structuring,summarization,use_case). Exists so the excluded mode can be "
                         "TESTED rather than assumed: `question` sits at 4.8%% contradicted (already "
                         "near the floor) and was never grounded in any run — it was the untouched "
                         "control that made the first widened A/B interpretable. The stated reason for "
                         "keeping it out, that grounding direct QA risks teaching refusal-ish hedging, "
                         "is UNTESTED. It is a plausible worry (refusals are 52%% of the deployed "
                         "chain's residual failures) but no arm has ever measured it. Use "
                         "--ground-modes question,structuring,summarization,use_case to run that arm.")
    ap.add_argument("--ground-keep-instruction", action="store_true",
                    help="also SAVE the grounding text in the stored question. Default is to store "
                         "the plain question, so the cartridge learns 'plain structuring request -> "
                         "grounded answer' rather than acquiring an instruction-conditioned policy "
                         "it will never be prompted with at inference (cf. the style-lock finding: "
                         "instruction text saved into questions becomes a conditional QA policy). "
                         "Set this only to A/B that choice.")
    ap.add_argument("--reanswer", default=None, metavar="FILE",
                    help="DECOUPLED M_A ARM: skip question generation entirely — read the questions "
                         "of an existing self-study jsonl and re-answer them with --model-id. Only "
                         "ONE model is ever resident, so an 8B-asks/4B-answers cell (or the reverse) "
                         "fits a single 24 GB card, which co-loading both cannot. It also makes the "
                         "M_A contrast exact: the paired cells share byte-identical questions, so "
                         "the delta is purely the answerer, with zero question-sampling noise.")
    ap.add_argument("--passage-ids", default=None)
    ap.add_argument("--tag-as", default=None,
                    help="override written passage_id for ALL records (pool under one cartridge)")
    ap.add_argument("--modes", default=None,
                    help="comma-sep seed modes to cycle (default: question,question,structuring,"
                         "summarization,use_case). Use 'relational' for the descriptive-addressability "
                         "pass (Phase 12-H): refers to the subject by its in-article descriptors.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seen-from", nargs="+", default=None, metavar="FILE",
                    help="EXTENSION ARM: pre-load each passage's ALREADY-GENERATED questions into its "
                         "dedup set, so every question this run produces is novel BY CONSTRUCTION. "
                         "Without it, a second run at a fresh seed regenerates the same canonical "
                         "questions (measured: a seed-999 probe collided 489/600 = 81%% with the "
                         "existing pool, because the `seen` set is per-run and question sampling is "
                         "temperature 0.6) — you would pay full generation cost for a half yield and "
                         "then discard the duplicates at merge time.")
    ap.add_argument("--stall-rounds", type=int, default=0, metavar="K",
                    help="stop a passage after K consecutive rounds that yield ZERO new questions "
                         "(0 = off). REQUIRED whenever --seen-from is used: a saturated passage would "
                         "otherwise burn its entire round budget (the loop allows up to --per-passage "
                         "rounds ~ 1 h/passage) producing nothing. The round at which a passage stalls "
                         "IS its self-study saturation point, which is the measurement this arm wants.")
    ap.add_argument("--saturation-out", default=None, metavar="FILE",
                    help="write per-passage {made, rounds, stalled, seeded} JSON here, so the "
                         "saturation curve is plottable without re-parsing logs. Rewritten after "
                         "EVERY passage, not just at the end, so it doubles as the resume ledger.")
    ap.add_argument("--resume", action="store_true",
                    help="APPEND to --out-file instead of truncating it, and skip passages already "
                         "finished (at target, or recorded as stalled in --saturation-out). A "
                         "half-finished passage tops up to target: its existing questions are folded "
                         "into the dedup set and its count carries over. This makes a long run "
                         "killable at any moment and resumable on ANOTHER machine — copy the "
                         "out-file + saturation sidecar across and re-run with the same flags. "
                         "Without it the generator truncates, so an interrupted run silently loses "
                         "everything and a partial file can masquerade as complete.")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",")] if args.modes else _MODES
    if args.ground_modes:                      # testable override; default leaves _GROUNDED_MODES alone
        global _GROUNDED_MODES
        _GROUNDED_MODES = {m.strip() for m in args.ground_modes.split(",") if m.strip()}
        print(f"--ground-modes: grounding applies to {sorted(_GROUNDED_MODES)} "
              f"(default is structuring, summarization, use_case)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    passages = load_passages(Path(args.passages_file))
    if args.passage_ids:
        wanted = {p.strip() for p in args.passage_ids.split(",")}
        passages = [p for p in passages if p["passage_id"] in wanted]
    print(f"Loaded {len(passages)} passages from {args.passages_file}")

    a_model, tokenizer = load_model(args.model_id, device)     # answer/target model
    if tokenizer.pad_token is None:                            # needed for left-padded batching
        tokenizer.pad_token = tokenizer.eos_token

    if args.reanswer:
        _reanswer(args, passages, a_model, tokenizer, device)
        return

    if args.question_model and args.question_model != args.model_id:
        q_model, q_tok = load_model(args.question_model, device)
        if q_tok.pad_token is None:
            q_tok.pad_token = q_tok.eos_token
        print(f"Decoupled question model: {args.question_model}")
    else:
        q_model, q_tok = a_model, tokenizer                    # same model writes + answers

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # EXTENSION ARM: questions already generated for each passage, keyed exactly like the in-run
    # dedup below (lowercased question text), so a pre-seeded passage can only produce novel output.
    prior: dict[str, set[str]] = {}
    if args.seen_from:
        for path in args.seen_from:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        prior.setdefault(r["passage_id"], set()).add(r["question"].lower())
        print(f"--seen-from: pre-loaded {sum(len(v) for v in prior.values())} questions "
              f"over {len(prior)} passages from {len(args.seen_from)} file(s)")
        if args.stall_rounds <= 0:
            print("  WARNING: --seen-from without --stall-rounds — a saturated passage will burn its "
                  "full round budget producing nothing. Recommend --stall-rounds 25.")

    # RESUME: fold anything already written back in, so a killed run continues instead of restarting.
    done: dict[str, dict] = {}          # passage_id -> its saturation record from a previous run
    carried: collections.Counter = collections.Counter()
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    prior.setdefault(r["passage_id"], set()).add(r["question"].lower())
                    carried[r["passage_id"]] += 1
        if args.saturation_out and Path(args.saturation_out).exists():
            done = {r["passage_id"]: r for r in json.loads(Path(args.saturation_out).read_text())}
        print(f"--resume: {sum(carried.values())} conversations already written over "
              f"{len(carried)} passages; {sum(1 for r in done.values() if r['stalled'] or r['made'] >= r['target'])} "
              f"passage(s) recorded finished")

    n_written = 0
    saturation: list[dict] = [done[p["passage_id"]] for p in passages
                              if p["passage_id"] in done]
    finished = {p for p, r in done.items() if r["stalled"] or r["made"] >= r["target"]}

    def flush_sidecar():
        if args.saturation_out:
            Path(args.saturation_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.saturation_out).write_text(json.dumps(saturation, indent=2))

    with open(out_path, "a" if args.resume else "w", encoding="utf-8") as f:
        for i, entry in enumerate(passages):
            pid, passage = entry["passage_id"], entry["context"]
            if pid in finished:
                print(f"\n[{i+1}/{len(passages)}] {pid} — already finished "
                      f"({done[pid]['made']} convs{', stalled' if done[pid]['stalled'] else ''}), skipping")
                continue
            print(f"\n[{i+1}/{len(passages)}] {pid} | {entry.get('title', pid)}")
            cached = (build_passage_prefix(a_model, tokenizer, passage, device)
                      if args.prefix_cache else None)
            seen: set[str] = set(prior.get(pid, ()))
            n_seeded = len(seen) - carried[pid]      # pre-existing corpus only, excl. this run's output
            made, round_i, dry_streak, stalled = carried[pid], 0, 0, False
            stats_parsed = stats_kept = 0
            if made:
                print(f"  resuming at {made}/{args.per_passage}")
            while made < args.per_passage and round_i < args.per_passage:  # bounded rounds
                # 1) multi-question generation. R>1 runs the next R rounds as ONE batched call:
                #    question generation is ~87% of pipeline cost at 2000/pp (248 sequential rounds x
                #    640 tokens per article) and was the only unbatched stage left. The R rounds cannot
                #    see each other's questions, so within-batch duplicates are filtered after the fact
                #    rather than prevented — correctness is unaffected (`seen` still gates every stored
                #    question), the cost is yield, which the sweep measures.
                R = max(1, min(args.question_batch_rounds, args.per_passage - round_i))
                batch_modes = [modes[(round_i + j) % len(modes)] for j in range(R)]
                round_i += R
                if R == 1:                       # byte-identical to the pre-batching path
                    raws = [_generate_one(q_model, q_tok,
                                          _build_conv_prefix(_multi_prompt(batch_modes[0],
                                                                           args.n_per_call),
                                                             passage, q_tok),
                                          args.q_max_tokens, args.q_temperature, device)]
                else:
                    raws = _generate_batch(q_model, q_tok,
                                           [_build_conv_prefix(_multi_prompt(m, args.n_per_call),
                                                               passage, q_tok) for m in batch_modes],
                                           args.q_max_tokens, args.q_temperature, device)
                # Seed mode is a per-ROUND property, so with R>1 a batch mixes modes and every
                # downstream decision (grounding, seed_type) must be per QUESTION, not per batch.
                assert len(raws) == len(batch_modes), (
                    f"generator returned {len(raws)} completions for {len(batch_modes)} rounds — "
                    f"zip would silently drop modes and mislabel seed_type")
                pairs = [(m, q) for m, r in zip(batch_modes, raws) for q in _parse_items(r)]
                stats_parsed += len(pairs)
                pairs = [(m, q) for m, q in pairs
                         if q.lower() not in seen and not seen.add(q.lower())]
                stats_kept += len(pairs)
                q_modes = [m for m, _ in pairs]
                questions = [q for _, q in pairs]
                if not questions:
                    # a batch consumes R rounds, so a dry batch must advance the streak by R —
                    # otherwise --stall-rounds 25 would silently mean "25 BATCHES" and the saturation
                    # ledger would stop being comparable to the 4000/pp run.
                    dry_streak += R
                    if args.stall_rounds and dry_streak >= args.stall_rounds:
                        stalled = True
                        print(f"  STALLED after {round_i} rounds — {args.stall_rounds} consecutive "
                              f"rounds with no new question. Saturation point for this passage.")
                        break
                    continue
                dry_streak = 0
                keep_n = args.per_passage - made
                questions, q_modes = questions[:keep_n], q_modes[:keep_n]   # keep them aligned
                if args.terse_frac > 0:
                    questions = [q + SHORT if random.random() < args.terse_frac else q
                                 for q in questions]
                # ASKED text may carry the grounding constraint; STORED text is the plain question
                # unless --ground-keep-instruction, so training data is not conditioned on an
                # instruction the cartridge never sees at inference. Decided PER QUESTION: with R>1 a
                # batch mixes seed modes, and grounding applies only to _GROUNDED_MODES.
                gflag = [args.ground_structured and m in _GROUNDED_MODES for m in q_modes]
                asked = [q + GROUND if g else q for q, g in zip(questions, gflag)]
                stored = [a if (g and args.ground_keep_instruction) else q
                          for q, a, g in zip(questions, asked, gflag)]
                # 2) batched greedy answers from the TARGET model
                for b in range(0, len(questions), args.answer_batch_size):
                    chunk = asked[b:b + args.answer_batch_size]
                    prefixes = [_build_conv_prefix(q, passage, tokenizer) for q in chunk]
                    answers = gen_answers(a_model, tokenizer, prefixes, args.a_max_tokens,
                                          args.answer_temperature, device, cached)
                    for q, m, a in zip(stored[b:b + args.answer_batch_size],
                                       q_modes[b:b + args.answer_batch_size], answers):
                        if not a:
                            continue
                        f.write(json.dumps({"passage_id": args.tag_as or pid, "seed_type": m,
                                            "question": q, "answer": a}, ensure_ascii=False) + "\n")
                        made += 1
                        n_written += 1
                f.flush()
            dup = (1 - stats_kept / stats_parsed) * 100 if stats_parsed else 0.0
            print(f"  wrote {made} conversations ({round_i} calls"
                  f"{f', {dup:.0f}% dup-waste' if args.question_batch_rounds > 1 else ''}"
                  f"{f', {n_seeded} pre-seen' if n_seeded else ''}"
                  f"{', STALLED' if stalled else ''})")
            saturation.append({"passage_id": pid, "made": made, "rounds": round_i,
                               "stalled": stalled, "seeded": n_seeded,
                               "target": args.per_passage,
                               "parsed": stats_parsed, "kept": stats_kept,
                               "qbatch": args.question_batch_rounds})
            flush_sidecar()          # after EVERY passage: this is the resume ledger

    print(f"\n{'='*60}\nWrote {n_written} conversations this run → {out_path}")
    if saturation:
        made_all = sorted(s["made"] for s in saturation)
        med = made_all[len(made_all) // 2]
        n_stalled = sum(s["stalled"] for s in saturation)
        print(f"per-passage novel: min {made_all[0]} · median {med} · max {made_all[-1]} "
              f"· target {args.per_passage} · stalled {n_stalled}/{len(saturation)} "
              f"· passages done {len(saturation)}/{len(passages)}")
        flush_sidecar()
        if args.saturation_out:
            print(f"saturation sidecar → {args.saturation_out}")


if __name__ == "__main__":
    main()
