"""
cartridge_qwen.py — Qwen3 cartridge pipeline

Milestones:
  M1  load_model, format_chat_*, compute_perplexity   — model + QA perplexity
  M2  QwenCartridge, CartridgeCache                   — KV prefix init + injection
  M4  QwenCartridge.train_distillation                — context distillation loop
"""
from __future__ import annotations

import math
import random
import argparse
import contextlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)


# ── Device ──────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── M1: Model loading ────────────────────────────────────────────────────────────

def load_model(model_id: str = "Qwen/Qwen3-1.7B", device: str | None = None,
               load_in_8bit: bool = False):
    """
    Load Qwen3 model and tokenizer in bfloat16.
    `load_in_8bit` (bitsandbytes) is for the case where a SECOND, larger model must be co-resident
    with the 4B cartridge reader — e.g. an 8B agent policy alongside 4B+cartridges on one 24 GB GPU
    (16.4 + 8.0 GB in bf16 would OOM; 8.5 + 8.0 fits). Inference is ~2-3x slower when quantized.
    Returns (model, tokenizer).
    """
    if device is None:
        device = get_device()
    print(f"Loading {model_id} → {device}{' [8-bit]' if load_in_8bit else ''} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
            trust_remote_code=True,
        )
    elif device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)

    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {n / 1e9:.2f}B params  dtype={next(model.parameters()).dtype}")
    return model, tokenizer


# ── M1: Chat-template formatting ─────────────────────────────────────────────────

def _build_prefix(question: str, passage: str | None, tokenizer) -> str:
    """
    Build the chat-formatted prefix (everything before the answer).

    With passage=None  → student format: "Question: {q}" as user turn
    With passage=text  → teacher format: "Context: {passage}\n\nQuestion: {q}"

    enable_thinking=False suppresses <think> tokens in Qwen3-Instruct so the
    model goes straight to its answer, matching the plain-answer perplexity target.
    """
    if passage is not None:
        content = f"Context: {passage.strip()}\n\nQuestion: {question}"
    else:
        content = f"Question: {question}"

    messages = [{"role": "user", "content": content}]
    prefix = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return prefix + "Short answer: "


# System instruction placed before the passage in the teacher's context. Kept short
# and generic so the cartridge (which replaces the passage) does not need to encode it.
_CONV_SYSTEM = "Use the following passage to answer.\n\n{passage}"


def _build_conv_prefix(question: str, passage: str | None, tokenizer) -> str:
    """
    Conversational prefix for SELF-STUDY (no "Short answer:" suffix).

    Teacher (passage given): system message holds the passage, user message holds
        the question/request.
    Student (passage=None):  only the user question; the cartridge supplies the
        passage knowledge.

    This same builder is used both when GENERATING the answer (turn B) and when
    computing the distillation teacher prefix, so the teacher force-decodes its own
    generation distribution (sharp, confident KL targets).
    """
    if passage is not None:
        messages = [
            {"role": "system", "content": _CONV_SYSTEM.format(passage=passage.strip())},
            {"role": "user", "content": question},
        ]
    else:
        messages = [{"role": "user", "content": question}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_with_answer(prefix_text: str, answer: str, tokenizer) -> tuple[list[int], int]:
    """
    Tokenize `prefix_text + answer` and return (full_ids, ans_start).

    ans_start is computed from the prefix token count — no BPE boundary ambiguity
    because we know exactly where the answer begins in the token sequence.
    """
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    ans_ids = tokenizer.encode(answer, add_special_tokens=False)
    full_ids = prefix_ids + ans_ids
    return full_ids, len(prefix_ids)


def crop_context(context: str, answer_start_char: int, answer: str, window_chars: int = 600) -> str:
    """
    Return a substring of `context` centred on the answer.

    Uses SQuAD's character-level answer_start to locate the answer, then
    extends a window of ~window_chars in each direction. This keeps the teacher
    input within a predictable token budget regardless of passage length.
    """
    answer_end = answer_start_char + len(answer)
    ctx_start = max(0, answer_start_char - window_chars // 2)
    ctx_end = min(len(context), answer_end + window_chars // 2)
    excerpt = context[ctx_start:ctx_end].strip()
    return excerpt


# ── M1: Perplexity ───────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    question: str,
    answer: str,
    cartridge: "QwenCartridge | ComposedCartridge | None" = None,
    passage: str | None = None,
    device: str = "cpu",
    cartridge_temp: float = 1.0,
) -> float:
    """
    Answer-token perplexity for one QA pair.

    cartridge=None, passage=None → baseline (no context)
    cartridge=<cart>             → student condition (single cartridge KV prefix)
    cartridge=<ComposedCartridge>→ gated multi-cartridge composition (>=2 cartridges)
    passage=<text>               → teacher condition (full context in input)
    """
    prefix_text = _build_prefix(question, passage, tokenizer)
    full_ids, ans_start = encode_with_answer(prefix_text, answer, tokenizer)

    if ans_start <= 0 or ans_start >= len(full_ids):
        return float("nan")

    inp_ids = full_ids[:-1]   # input: all but last token
    tgt_ids = full_ids[1:]    # target: all but first token

    inp = torch.tensor(inp_ids, dtype=torch.long, device=device).unsqueeze(0)
    tgt = torch.tensor(tgt_ids, dtype=torch.long, device=device)

    if isinstance(cartridge, ComposedCartridge) and cartridge.mode == "concat":
        cache = StackedCartridgeCache(cartridge.components, model, device=device, dtype=model.dtype)
        cart_len = len(cartridge.components) * cartridge.cartridge_len  # query after all blocks
        pos_ids = torch.arange(
            cart_len, cart_len + inp.shape[1], dtype=torch.long, device=device
        ).unsqueeze(0)
        outputs = model(input_ids=inp, past_key_values=cache, position_ids=pos_ids, use_cache=True)
    elif isinstance(cartridge, ComposedCartridge):
        cart_len = cartridge.cartridge_len
        cache = ComposedCartridgeCache(
            cartridge.components, device=device, dtype=model.dtype,
            cartridge_temp=cartridge_temp,
        )
        pos_ids = torch.arange(
            cart_len, cart_len + inp.shape[1], dtype=torch.long, device=device
        ).unsqueeze(0)
        with composed_attention(model, cache):
            outputs = model(
                input_ids=inp,
                past_key_values=cache,
                position_ids=pos_ids,
                use_cache=True,
            )
    elif cartridge is not None:
        cart_len = cartridge.cartridge_len
        cache = CartridgeCache(cartridge, device=device, dtype=model.dtype)
        pos_ids = torch.arange(
            cart_len, cart_len + inp.shape[1], dtype=torch.long, device=device
        ).unsqueeze(0)
        outputs = model(
            input_ids=inp,
            past_key_values=cache,
            position_ids=pos_ids,
            use_cache=True,
        )
    else:
        outputs = model(input_ids=inp, use_cache=False)

    logits = outputs.logits  # (1, T, vocab)

    # Position predicting the first answer token = ans_start - 1
    pred_start = ans_start - 1
    if pred_start >= logits.shape[1]:
        return float("nan")

    loss = F.cross_entropy(logits[0, pred_start:], tgt[pred_start:], reduction="mean")
    return math.exp(loss.item())


# ── M5: Greedy generation (for EM / F1 / containment eval) ─────────────────────────

@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    question: str,
    cartridge: "QwenCartridge | ComposedCartridge | None" = None,
    passage: str | None = None,
    device: str = "cpu",
    max_new_tokens: int = 32,
    conversational: bool = False,
    cartridge_temp: float = 1.0,
    concat_shared_sink: bool = False,
) -> str:
    """
    Greedily decode an answer for one question under one condition, returning the
    first decoded line (stripped).

      cartridge=None, passage=None → baseline (no context)
      cartridge=<cart>             → cartridge KV prefix (RoPE-offset positions)
      passage=<text>               → teacher (full context in the prompt)

    conversational=False → `_build_prefix` (extractive "Short answer: " cue).
    conversational=True  → `_build_conv_prefix` (the SELF-STUDY chat format the
        cartridge is actually trained under: system holds the passage, user holds
        the question, no "Short answer:" suffix). A cartridge trained with
        full-conversation KL terminates immediately under the extractive prompt
        (out of distribution), so it MUST be evaluated under this format; answers
        come back as full sentences and containment scoring handles them.

    The decode loop reuses the validated single-forward mechanics of
    compute_perplexity (cartridge cache + position_ids starting at cartridge_len),
    advancing one token at a time, so generation with a cartridge stays consistent
    with how that cartridge is scored.
    """
    builder = _build_conv_prefix if conversational else _build_prefix
    prefix_text = builder(question, passage, tokenizer)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    gated = False
    if isinstance(cartridge, ComposedCartridge):
        if cartridge.mode == "concat":
            cache = StackedCartridgeCache(cartridge.components, model, device=device, dtype=model.dtype,
                                          shared_sink=concat_shared_sink, tokenizer=tokenizer)
            cart_len = (1 if concat_shared_sink else 0) + \
                len(cartridge.components) * cartridge.cartridge_len  # query after sink + all blocks
        else:
            # gated (relevance-gate + 50/50) OR gnorm (joint-softmax, in-distribution) —
            # both attend each cartridge at its own positions via the patched attention,
            # query at cart_len; the cache.mode selects the combiner in _composed_attn.
            cache = ComposedCartridgeCache(
                cartridge.components, device=device, dtype=model.dtype,
                cartridge_temp=cartridge_temp, mode=cartridge.mode,
                shared_sink=getattr(cartridge, "shared_sink", False),
                model=model, tokenizer=tokenizer,
                gate=getattr(cartridge, "gate", None),
            )
            cart_len = cartridge.cartridge_len
            gated = True
    elif cartridge is not None:
        cart_len = cartridge.cartridge_len
        cache = CartridgeCache(cartridge, device=device, dtype=model.dtype)
    else:
        cart_len = 0
        cache = DynamicCache()

    # Tokens that terminate generation: eos plus the chat turn-end token.
    stop_ids = {tokenizer.eos_token_id}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        stop_ids.add(im_end)
    newline_ids = set(tokenizer.encode("\n", add_special_tokens=False))

    cur = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    n_done = 0          # real (non-cartridge) tokens already in the cache
    generated: list[int] = []

    # Gated composition must keep its attention patch active for every decode step;
    # concat composition uses the stock attention, so no context is needed.
    loop_ctx = composed_attention(model, cache) if gated else contextlib.nullcontext()
    with loop_ctx:
        for _ in range(max_new_tokens):
            inp_len = cur.shape[1]
            pos = torch.arange(
                cart_len + n_done, cart_len + n_done + inp_len,
                dtype=torch.long, device=device,
            ).unsqueeze(0)
            out = model(
                input_ids=cur,
                past_key_values=cache,
                position_ids=pos,
                use_cache=True,
                logits_to_keep=1,   # greedy decode needs only the last-token logits
            )
            nxt = int(out.logits[0, -1].argmax())
            n_done += inp_len
            # Hard stop on end-of-turn always; on a newline only once content has begun.
            # A leading newline/whitespace (common right after the assistant header) is
            # skipped rather than terminating, so a single stray token can't zero the
            # whole prediction — important for cartridges that emit a leading newline.
            if nxt in stop_ids:
                break
            if nxt in newline_ids:
                if generated:
                    break
                cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
                continue
            generated.append(nxt)
            cur = torch.tensor([[nxt]], dtype=torch.long, device=device)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip().split("\n")[0].strip()


# ── M2: CartridgeCache ───────────────────────────────────────────────────────────

class CartridgeCache(DynamicCache):
    """
    DynamicCache pre-filled with a QwenCartridge's K,V tensors.

    At init: DynamicCache.update() is called once per layer, storing the cartridge
    K,V as the initial key_cache[l] / value_cache[l] entries.

    When the model's attention layer calls cache.update(new_k, new_v, l) during
    forward, DynamicCache concatenates the stored cartridge K,V with new_k, new_v
    via torch.cat — which preserves the gradient path back to cartridge.keys and
    cartridge.values during training.

    Query tokens must use position_ids starting at cartridge_len so RoPE sees
    them as occurring after the cartridge prefix (positions 0..cartridge_len-1).

    The cartridge parameters are kept in float32 (good for the optimizer), but
    the model runs in bfloat16. We cast K,V to `dtype` at injection so they match
    the model's compute dtype; autograd routes gradients back through the cast to
    the fp32 parameters, so training is unaffected.
    """

    def __init__(self, cartridge: "QwenCartridge", device: str = "cpu", dtype=None):
        super().__init__()
        for l in range(cartridge.num_layers):
            k = cartridge.keys[l].unsqueeze(0).to(device=device, dtype=dtype)    # (1, kv_heads, cart_len, head_dim)
            v = cartridge.values[l].unsqueeze(0).to(device=device, dtype=dtype)
            self.update(k, v, l)


def _sink_kv(model, tokenizer, n_layers, device, dtype):
    """KV for ONE neutral attention-sink position — the KV the model naturally places at
    position 0 for a sequence-start token (RoPE-baked at position 0). Prepended before the
    composed cartridges so the model's first-token attention magnet lands on this neutral
    slot instead of the first cartridge's (content-bearing) slot 0."""
    tid = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if not isinstance(tid, int) or tid < 0:
        tid = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
    ids = torch.tensor([[tid]], dtype=torch.long, device=device)
    base = getattr(model, "model", model)
    with torch.no_grad():
        past = base(input_ids=ids, use_cache=True).past_key_values
    out = []
    for l in range(n_layers):
        k, v = _layer_kv(past, l)
        out.append((k.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype)))
    return out


class StackedCartridgeCache(DynamicCache):
    """KV-concatenation composition at DISTINCT positions (fixes the overlapping-position
    OOD failure where compose([B,B]) < cart_B). Component k is placed at its OWN position
    block: its keys are RoPE-baked at phase 0..cart_len-1, so we re-RoPE component k by a
    uniform shift (R(i+s)=R(s)R(i) ⇒ one extra rotation R(s) per key). Values are position-
    independent (RoPE touches keys only). The query sits after all blocks, so the LAST
    component lands at its trained relative offset cart_len..1 (in-distribution) — order the
    answer-relevant cartridge last.

    shared_sink=True prepends ONE neutral sink position (a sequence-start token's KV) at
    position 0 and shifts every component by +1 (block k → [1+k*cart_len ..]). This keeps
    the model's first-token attention magnet on a neutral slot rather than on the first
    cartridge's content — the fix for compose([A,B]) ≪ cart_B destructive interference,
    where the non-answer cartridge's slot 0 (the sink) was injecting its content."""

    def __init__(self, components: list["QwenCartridge"], model, device: str = "cpu", dtype=None,
                 shared_sink: bool = False, tokenizer=None):
        super().__init__()
        n_layers = components[0].num_layers
        cart_len = components[0].cartridge_len
        rotary = getattr(getattr(model, "model", model), "rotary_emb", None)
        sink = _sink_kv(model, tokenizer, n_layers, device, dtype) if shared_sink else None
        off = 1 if sink is not None else 0   # +1 position for the prepended sink
        for l in range(n_layers):
            ks, vs = [], []
            if sink is not None:
                ks.append(sink[l][0])
                vs.append(sink[l][1])
            for j, c in enumerate(components):
                k = c.keys[l].unsqueeze(0).to(device=device, dtype=dtype)   # (1, kv_heads, cart_len, hd)
                v = c.values[l].unsqueeze(0).to(device=device, dtype=dtype)
                shift = off + j * cart_len
                if shift and rotary is not None:
                    pos = torch.full((1, cart_len), shift, dtype=torch.long, device=device)
                    cos, sin = rotary(k, pos)                 # R(shift) for every slot
                    _, k = apply_rotary_pos_emb(k, k, cos, sin)
                ks.append(k)
                vs.append(v)
            self.update(torch.cat(ks, dim=2), torch.cat(vs, dim=2), l)


# ── Cartridge composition (multihop / cross-document) ─────────────────────────────
#
# Composing N>=2 cartridges does NOT concatenate their K/V into one prefix — that
# would place later cartridges at RoPE positions cart_len..N*cart_len-1, out of the
# distribution every cartridge was trained at (queries always sit one cart_len after
# a prefix occupying positions 0..cart_len-1).
#
# Instead each component is attended INDEPENDENTLY at its own positions 0..cart_len-1
# (separate softmax per cartridge), the per-cartridge outputs are combined by a
# relevance gate, and that pooled cartridge output is blended 50/50 with ordinary
# causal self-attention over the live prompt context. This is a faithful port of the
# TinyStories gated path (train_cartridge.py: attn_t.forward, multi_cartridge_kv).
#
# A single component (compose([c]) -> c) uses the normal single-cartridge path; the
# gated path only activates for >=2 components.


class ComposedCartridge:
    """Lightweight container marking a set of equal-length cartridges to be composed
    at inference. Exposes `cartridge_len` so callers share the single-cartridge
    RoPE-offset logic unchanged.

    mode="concat" (default): KV-concatenation — both cartridges' K/V go into ONE cache
        and the stock attention does a single joint softmax over all of them (the
        paper's mechanism; query offset stays cart_len since every cartridge keeps its
        trained positions 0..cart_len-1).
    mode="gated":  per-cartridge independent attention + relevance gate + 50/50 blend
        (the original path; dilutes the answer-bearing cartridge — kept for comparison)."""

    def __init__(self, components: list["QwenCartridge"], mode: str = "concat",
                 shared_sink: bool = False, gate=None):
        self.mode = mode
        self.shared_sink = shared_sink   # gnorm: one shared neutral sink instead of N per-cartridge
        self.gate = gate                 # gnorm: optional RelevanceGate (learned per-cart weight)
        ref = components[0]
        for c in components[1:]:
            if (c.num_layers != ref.num_layers or c.num_kv_heads != ref.num_kv_heads
                    or c.head_dim != ref.head_dim):
                raise ValueError(
                    f"Cartridge shape mismatch: "
                    f"({c.num_layers},{c.num_kv_heads},?,{c.head_dim}) vs "
                    f"({ref.num_layers},{ref.num_kv_heads},?,{ref.head_dim})"
                )
            if c.cartridge_len != ref.cartridge_len:
                raise ValueError(
                    f"Composition requires equal cartridge lengths "
                    f"(got {c.cartridge_len} vs {ref.cartridge_len}). "
                    f"Retrain with matching --cartridge-len."
                )
        self.components = list(components)
        self.cartridge_len = ref.cartridge_len
        self.num_layers = ref.num_layers


def compose(cartridges: list["QwenCartridge"], mode: str = "concat",
            shared_sink: bool = False, gate=None) -> "QwenCartridge | ComposedCartridge":
    """Equal-length composition. One cartridge -> itself (normal single path);
    several -> a ComposedCartridge (mode="concat" KV-concatenation, or "gated"/"gnorm").
    shared_sink (gnorm only): use ONE shared neutral sink instead of each cartridge's own
    slot-0 sink. gate (gnorm only): a RelevanceGate that adds a learned per-cartridge weight
    bias, so distractors are downweighted (Phase 11 S1)."""
    if not cartridges:
        raise ValueError("Must provide at least one cartridge to compose")
    if len(cartridges) == 1:
        return cartridges[0]
    return ComposedCartridge(cartridges, mode=mode, shared_sink=shared_sink, gate=gate)


class RelevanceGate(nn.Module):
    """Learned per-cartridge relevance gate for gnorm composition (Phase 11 S1).

    gnorm's default weight is softmax over per-group logsumexp(q·k) — it measures attention
    MASS and is ~content-blind across similarly-trained cartridges (a distractor gets ~the same
    weight as the answer cartridge). This gate adds a LEARNED relevance bias per cartridge: it
    projects the query and the cartridge's keys into a low-rank relevance subspace (per layer),
    and the logsumexp of their attention there is added to that cartridge's weight-logit. Trained
    GATE-ONLY (cartridges frozen) so a distractor's bias goes low and a matching cartridge's high.
    Shared per-layer params → generalizes to unseen cartridges (the continual-learning property).
    Only the combination WEIGHTS change; per-cartridge attention outputs are untouched."""

    def __init__(self, num_layers: int, head_dim: int, rank: int = 32):
        super().__init__()
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.rank = rank
        s = head_dim ** -0.5
        self.Pq = nn.Parameter(torch.randn(num_layers, head_dim, rank) * s)
        self.Pk = nn.Parameter(torch.randn(num_layers, head_dim, rank) * s)

    def bias(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Relevance bias for one group. q:(b,h,T,hd) k:(b,h,L,hd) -> (b,h,T)."""
        pq = self.Pq[layer_idx].to(q.dtype)
        pk = self.Pk[layer_idx].to(q.dtype)
        qp = torch.matmul(q, pq)                              # (b,h,T,r)
        kp = torch.matmul(k, pk)                              # (b,h,L,r)
        logits = torch.matmul(qp, kp.transpose(-2, -1)) * (self.rank ** -0.5)
        return logits.float().logsumexp(dim=-1)               # (b,h,T)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "num_layers": self.num_layers,
                    "head_dim": self.head_dim, "rank": self.rank}, path)

    @staticmethod
    def load(path, device: str = "cpu") -> "RelevanceGate":
        ck = torch.load(path, map_location=device, weights_only=False)
        g = RelevanceGate(ck["num_layers"], ck["head_dim"], ck["rank"])
        g.load_state_dict(ck["state_dict"])
        return g.to(device)


class ComposedCartridgeCache(DynamicCache):
    """DynamicCache that holds ONLY the live prompt context (one entry per layer,
    grown by update() exactly like a vanilla DynamicCache). The component cartridge
    K/V are stashed separately in `components_kv` and consumed by the patched Qwen3
    attention, never concatenated into the context cache."""

    def __init__(self, components: list["QwenCartridge"], device: str = "cpu",
                 dtype=None, cartridge_temp: float = 1.0, debug: bool = False,
                 mode: str = "gated", shared_sink: bool = False, model=None, tokenizer=None,
                 gate=None):
        super().__init__()
        self.cartridge_temp = cartridge_temp
        self.debug = debug
        self.mode = mode           # "gated" (relevance-gate+50/50) | "gnorm" (joint softmax)
        self.gate = gate           # optional RelevanceGate: learned per-cartridge weight bias
        self.num_layers = components[0].num_layers
        self.cart_len = components[0].cartridge_len
        # gnorm shared sink: build ONE neutral sink (RoPE phase 0) and STRIP each cartridge's
        # own slot-0, so composing N cartridges yields ONE sink instead of N competing sinks
        # that flatten the joint-softmax gate. (No-op for concat/gated; needs model+tokenizer.)
        self.shared_sink = shared_sink
        self.sink_kv = (_sink_kv(model, tokenizer, self.num_layers, device, dtype)
                        if shared_sink else None)
        s0 = 1 if shared_sink else 0     # drop position 0 of each cartridge when sink is shared
        # Per layer: list of (k_i, v_i), each (1, num_kv_heads, cart_len[-1], head_dim).
        # Cartridge keys are already k_norm'd + RoPE-baked (extracted post-norm/rope at
        # init), so they are used raw here — matching the single-cartridge path.
        self.components_kv: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        for l in range(self.num_layers):
            layer = []
            for c in components:
                ck = c.keys[l][:, s0:, :].unsqueeze(0).to(device=device, dtype=dtype)
                cv = c.values[l][:, s0:, :].unsqueeze(0).to(device=device, dtype=dtype)
                layer.append((ck, cv))
            self.components_kv.append(layer)
        # debug gate-weight accumulation (mean over layers, per forward)
        self.last_cartridge_weights: torch.Tensor | None = None
        self._weight_acc: torch.Tensor | None = None
        # per-(layer, cartridge, head) weights at the LAST query position = the answer-generation
        # position (the one that produces the next token). The mean over ALL positions/heads
        # (above) washes out the gate's effect; this keeps head detail at the decisive position so
        # a top-quartile can surface the specific heads doing the hop. (Phase 11 gate diagnostic.)
        self.last_genpos_weights: torch.Tensor | None = None
        self._genpos_acc: list | None = None


# Active composed cache for the current forward; None means "behave normally".
_ACTIVE_COMPOSED_CACHE: "ComposedCartridgeCache | None" = None


@contextlib.contextmanager
def composed_attention(model, cache: "ComposedCartridgeCache"):
    """Activate the gated multi-cartridge attention path for the duration of a forward
    (or a decode loop). Restores the previous state on exit."""
    global _ACTIVE_COMPOSED_CACHE
    prev = _ACTIVE_COMPOSED_CACHE
    _ACTIVE_COMPOSED_CACHE = cache
    try:
        yield
    finally:
        _ACTIVE_COMPOSED_CACHE = prev


def _gnorm_attn(module, q, k_ctx, v_ctx, cache) -> torch.Tensor:
    """Global-normalized, in-distribution composition (the fix for concat's OOD dilution).

    Each cartridge is attended at its OWN in-distribution positions 0..cart_len-1 (query
    already at the cart_len offset — no re-RoPE, so EVERY cartridge sees the query at its
    trained relative offset, unlike distinct-position concat where non-last cartridges are
    OOD). The per-cartridge outputs AND the live-context output are then combined by ONE
    joint softmax over all their keys, via the group decomposition:
        joint = Σ_g (Z_g / Z) · out_g,   weight_g = softmax_g(logsumexp_g)   per (pos, head)
    where out_g = softmax_within_g(q·k)·v and Z_g = Σ exp(q·k) over group g's positions.
    This is mathematically the joint softmax over [all cartridges ∪ context] with each
    cartridge in-distribution — order-invariant, and the cartridge-vs-context balance is
    data-dependent (replacing the gated path's fixed 50/50). Returns (b, T, n_heads, hd)."""
    b, n_heads, T, hd = q.shape
    n_rep = module.num_key_value_groups
    scale = module.scaling

    outs, lses = [], []
    gate = getattr(cache, "gate", None)   # Phase 11 S1: learned per-cartridge relevance bias
    # Per cartridge, in-distribution: independent softmax → output + its logsumexp mass.
    for (ck, cv) in cache.components_kv[module.layer_idx]:
        ck = repeat_kv(ck, n_rep).expand(b, -1, -1, -1)   # (b, n_heads, cart_len, hd)
        cv = repeat_kv(cv, n_rep).expand(b, -1, -1, -1)
        logits_i = torch.matmul(q, ck.transpose(-2, -1)) * scale   # (b, n_heads, T, cart_len)
        outs.append(torch.matmul(
            torch.softmax(logits_i, dim=-1, dtype=torch.float32).to(q.dtype), cv))
        wl = logits_i.float().logsumexp(dim=-1)                     # (b, n_heads, T)  content mass
        if gate is not None:                                       # + learned relevance bias
            wl = wl + gate.bias(module.layer_idx, q, ck)
        lses.append(wl)

    # Shared neutral sink as ONE group (its slot-0 was stripped from every cartridge above),
    # so the cartridge groups' logsumexp reflects CONTENT and the joint softmax can downweight
    # irrelevant cartridges instead of flattening (each cartridge's own sink inflating its mass).
    if getattr(cache, "shared_sink", False) and getattr(cache, "sink_kv", None) is not None:
        sk, sv = cache.sink_kv[module.layer_idx]
        sk = repeat_kv(sk, n_rep).expand(b, -1, -1, -1)
        sv = repeat_kv(sv, n_rep).expand(b, -1, -1, -1)
        logits_s = torch.matmul(q, sk.transpose(-2, -1)) * scale   # (b, n_heads, T, 1)
        outs.append(torch.matmul(
            torch.softmax(logits_s, dim=-1, dtype=torch.float32).to(q.dtype), sv))
        lses.append(logits_s.float().logsumexp(dim=-1))            # (b, n_heads, T)

    # Live context (causal), computed manually so we get its logsumexp mass too.
    kc = repeat_kv(k_ctx, n_rep)
    vc = repeat_kv(v_ctx, n_rep)
    logits_c = torch.matmul(q, kc.transpose(-2, -1)) * scale       # (b, n_heads, T, T_ctx)
    nq, nk = q.size(2), kc.size(2)
    if nq == nk:
        neg = torch.triu(torch.ones(nq, nk, dtype=torch.bool, device=q.device), diagonal=1)
        logits_c = logits_c.masked_fill(neg, float("-inf"))
    elif nq != 1:                                                  # partial prefill w/ cache
        T0 = nk - nq
        neg = torch.zeros(nq, nk, dtype=torch.bool, device=q.device)
        neg[:, T0:] = torch.triu(torch.ones(nq, nq, dtype=torch.bool, device=q.device), diagonal=1)
        logits_c = logits_c.masked_fill(neg, float("-inf"))
    outs.append(torch.matmul(
        torch.softmax(logits_c, dim=-1, dtype=torch.float32).to(q.dtype), vc))
    lses.append(logits_c.float().logsumexp(dim=-1))                # (b, n_heads, T)

    # ONE joint softmax over all groups (cartridges + context), per (position, head).
    weights = torch.softmax(torch.stack(lses, dim=0), dim=0).to(q.dtype)   # (G, b, n_heads, T)
    out_all = torch.stack(outs, dim=0)                                     # (G, b, n_heads, T, hd)
    y = (weights.unsqueeze(-1) * out_all).sum(0)                           # (b, n_heads, T, hd)

    if cache.debug:                       # per-cartridge attention mass (mean pos/head/layer)
        nc = len(cache.components_kv[module.layer_idx])
        w_cart = weights[:nc].mean(dim=(1, 2, 3))
        contrib = w_cart.detach() / cache.num_layers
        cache._weight_acc = contrib if cache._weight_acc is None else cache._weight_acc + contrib
        # + per-(cartridge, head) weight at the LAST position (answer-generation position), per layer
        genpos = weights[:nc, 0, :, -1].detach().float()          # (nc, n_heads)
        cache._genpos_acc = [genpos] if cache._genpos_acc is None else cache._genpos_acc + [genpos]
        if module.layer_idx == cache.num_layers - 1:
            cache.last_cartridge_weights = cache._weight_acc.unsqueeze(0)
            cache.last_genpos_weights = torch.stack(cache._genpos_acc, dim=0)  # (n_layers, nc, heads)

    return y.transpose(1, 2).contiguous()


def _composed_attn(module, q, k_ctx, v_ctx, cache) -> torch.Tensor:
    """Gated cartridge attention + causal context attention, blended 50/50.

    q, k_ctx, v_ctx are post-RoPE/post-norm states (q already at the cart_len offset).
    Returns (b, T, n_heads, head_dim) to match Qwen3Attention's reshape expectation.
    """
    if getattr(cache, "mode", "gated") == "gnorm":
        return _gnorm_attn(module, q, k_ctx, v_ctx, cache)

    b, n_heads, T, hd = q.shape
    n_rep = module.num_key_value_groups
    scale = module.scaling

    # Reset the per-forward debug accumulator at the first layer.
    if cache.debug and module.layer_idx == 0:
        cache._weight_acc = None

    # ── Per-cartridge: independent softmax → output + relevance score ──────────────
    cart_attns, cart_scores = [], []
    for (ck, cv) in cache.components_kv[module.layer_idx]:
        ck = repeat_kv(ck, n_rep).expand(b, -1, -1, -1)   # (b, n_heads, cart_len, hd)
        cv = repeat_kv(cv, n_rep).expand(b, -1, -1, -1)
        logits_i = torch.matmul(q, ck.transpose(-2, -1)) * scale   # (b, n_heads, T, cart_len)
        attn_i = torch.matmul(
            torch.softmax(logits_i, dim=-1, dtype=torch.float32).to(q.dtype), cv
        )                                                          # (b, n_heads, T, hd)
        # Best-matching query position speaks for the cartridge (max over T), mean heads.
        score_i = (logits_i.float().logsumexp(dim=-1)
                   - math.log(ck.size(-2))).max(dim=-1).values.mean(dim=-1)  # (b,)
        cart_attns.append(attn_i)
        cart_scores.append(score_i)

    cart_weights = torch.softmax(
        torch.stack(cart_scores, dim=-1) / cache.cartridge_temp, dim=-1
    )                                                              # (b, n_components)
    all_cart = torch.stack(cart_attns, dim=0)                      # (n, b, n_heads, T, hd)
    w = cart_weights.t()[:, :, None, None, None].to(all_cart.dtype)  # (n, b, 1,1,1)
    weighted_cart = (all_cart * w).sum(0)                          # (b, n_heads, T, hd)

    # ── Context self-attention (causal), shapes mirror the TinyStories port ────────
    kc = repeat_kv(k_ctx, n_rep)
    vc = repeat_kv(v_ctx, n_rep)
    nq, nk = q.size(2), kc.size(2)
    if nq == nk:
        out_ctx = F.scaled_dot_product_attention(q, kc, vc, is_causal=True, scale=scale)
    elif nq == 1:
        out_ctx = F.scaled_dot_product_attention(q, kc, vc, is_causal=False, scale=scale)
    else:
        mask = torch.zeros((nq, nk), dtype=torch.bool, device=q.device)
        T0 = nk - nq
        mask[:, :T0] = True
        mask[:, T0:] = torch.tril(torch.ones((nq, nq), dtype=torch.bool, device=q.device))
        out_ctx = F.scaled_dot_product_attention(q, kc, vc, attn_mask=mask, scale=scale)

    # ── Fixed 50/50 cartridge-vs-context blend (N-independent) ─────────────────────
    y = (weighted_cart + out_ctx) / 2

    if cache.debug:
        contrib = cart_weights.detach() / cache.num_layers
        cache._weight_acc = contrib if cache._weight_acc is None else cache._weight_acc + contrib
        if module.layer_idx == cache.num_layers - 1:
            cache.last_cartridge_weights = cache._weight_acc

    return y.transpose(1, 2).contiguous()


_ORIG_QWEN3_ATTN_FORWARD = Qwen3Attention.forward


def _qwen3_attn_forward_patched(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Drop-in for Qwen3Attention.forward. When a ComposedCartridgeCache is active it
    runs the gated multi-cartridge path; otherwise it defers to the stock forward, so
    single-cartridge / teacher / baseline behavior is byte-for-byte unchanged."""
    cache = _ACTIVE_COMPOSED_CACHE
    if cache is None:
        return _ORIG_QWEN3_ATTN_FORWARD(
            self, hidden_states, position_embeddings, attention_mask,
            past_key_values=past_key_values, cache_position=cache_position, **kwargs,
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_values is not None:
        k, v = past_key_values.update(
            k, v, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position}
        )

    attn_output = _composed_attn(self, q, k, v, cache)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


Qwen3Attention.forward = _qwen3_attn_forward_patched


# ── Teacher corpus-KV cache (large-corpus distillation) ───────────────────────────
#
# For a ~20k-token corpus, re-running the teacher forward over [corpus + question +
# answer] every training step re-encodes the identical corpus thousands of times.
# Instead we prefill the CONSTANT system/corpus block once and reuse its KV across
# every step (same pattern as benchmark_cost.py's teacher_cached). Per step the
# teacher then forwards only [question + answer] against the cached corpus KV.

def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _layer_kv(cache, layer_idx: int):
    """Return (key, value) tensors for one layer, across transformers DynamicCache
    versions: newest stores `cache.layers[l].keys/.values`; older exposed
    `cache.key_cache[l]/.value_cache[l]`; legacy was subscriptable `cache[l] -> (k, v)`."""
    layers = getattr(cache, "layers", None)
    if layers is not None:
        layer = layers[layer_idx]
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        if k is not None and v is not None:
            return k, v
    if getattr(cache, "key_cache", None) is not None:
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    kv = cache[layer_idx]
    return kv[0], kv[1]


@torch.no_grad()
def build_teacher_corpus_cache(model, tokenizer, passage_text: str, device: str):
    """Prefill the constant teacher system/corpus block once.

    Returns (snapshot, const_len): `snapshot` is a list of (key, value) tensors per
    layer for the cached positions; `const_len` is how many leading positions are
    cached. The constant block = the token prefix shared by two teacher conv prefixes
    with *different* questions (i.e. system+corpus and the user-turn header, up to
    where the question content begins) — found by token-level common prefix, which
    also dodges any BPE merge at the corpus/question boundary."""
    pa = _build_conv_prefix("alpha", passage_text, tokenizer)
    pb = _build_conv_prefix("omega beta", passage_text, tokenizer)
    ia = tokenizer.encode(pa, add_special_tokens=False)
    ib = tokenizer.encode(pb, add_special_tokens=False)
    const_len = _common_prefix_len(ia, ib)
    const_ids = ia[:const_len]
    inp = torch.tensor([const_ids], dtype=torch.long, device=device)
    # Use the base transformer (model.model) to get past_key_values WITHOUT running the
    # LM head — avoids materializing the (seq × vocab) logits we don't need here, which
    # is significant memory at long corpus / large model. Fall back to the full model if
    # the .model attribute isn't present. KV are identical either way.
    base = getattr(model, "model", model)
    out = base(input_ids=inp, use_cache=True)
    past = out.past_key_values
    n_layers = model.config.num_hidden_layers
    snapshot = []
    for l in range(n_layers):
        k, v = _layer_kv(past, l)
        snapshot.append((k.detach(), v.detach()))
    return snapshot, const_len


def _cache_from_snapshot(snapshot) -> DynamicCache:
    """Fresh DynamicCache seeded with the snapshot KV. The snapshot tensors are never
    mutated (attention appends out-of-place via torch.cat), so the same snapshot is
    safely reused across steps; build a new cache per step so appends don't accumulate."""
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(snapshot):
        cache.update(k, v, layer_idx)
    return cache


# ── M2: QwenCartridge ────────────────────────────────────────────────────────────

class QwenCartridge(nn.Module):
    """
    Learned KV prefix for one SQuAD passage.

    Shape: (num_layers, num_kv_heads, cartridge_len, head_dim)
    Qwen3-1.7B: (28, 8, cartridge_len, 128)
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        cartridge_len: int,
        head_dim: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.cartridge_len = cartridge_len
        self.head_dim = head_dim
        self.keys = nn.Parameter(
            torch.zeros(num_layers, num_kv_heads, cartridge_len, head_dim)
        )
        self.values = nn.Parameter(
            torch.zeros(num_layers, num_kv_heads, cartridge_len, head_dim)
        )

    # ── Initialization ──────────────────────────────────────────────────────────

    @staticmethod
    def init_from_passage(
        model,
        tokenizer,
        passage_text: str,
        cartridge_len: int,
        device: str,
    ) -> "QwenCartridge":
        """
        Run the frozen model on the first cartridge_len passage tokens and store
        the resulting past_key_values as the cartridge's initial K,V state.

        The extracted K tensors already have RoPE baked in at positions
        0..cartridge_len-1. At inference / training time, query tokens use
        position_ids starting at cartridge_len so attention scores are correct.
        """
        cfg = model.config
        num_layers = cfg.num_hidden_layers
        num_kv_heads = cfg.num_key_value_heads
        # Qwen3 decouples head_dim from hidden_size/num_heads (it's an explicit 128),
        # so read cfg.head_dim when present; the // fallback only coincidentally matched
        # for 1.7B (2048/16=128) and is wrong for 4B (2560/32=80 ≠ real 128).
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)

        # Run the model on the FULL passage so init reflects the whole document,
        # then select cartridge_len KV slots from it (rather than only the first
        # cartridge_len tokens, which left facts in the passage tail unseen).
        ids = tokenizer.encode(passage_text, add_special_tokens=False)
        passage_len = len(ids)

        inp = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            # Base transformer (no LM head) → past_key_values without materializing the
            # (seq × vocab) logits init never uses (~5 GB at 16.6k; OOMs 8B otherwise).
            base = getattr(model, "model", model)
            out = base(input_ids=inp, use_cache=True)

        past = out.past_key_values  # DynamicCache, seq_len == passage_len

        if passage_len == cartridge_len:
            sel = torch.arange(passage_len, device=device)
        elif passage_len < cartridge_len:
            # Short passage: PAD up to cartridge_len by cycling through the passage tokens
            # (contiguous 0..passage_len-1, then wrap) so ALL cartridges are exactly
            # cartridge_len and remain composable (equal-length requirement). The wrapped
            # slots carry their original RoPE phase — an init-only approximation, like the
            # long-passage striding below; distillation reshapes the KV and query
            # position_ids (starting at cart_len) are unchanged.
            print(f"  Passage ({passage_len}) < cartridge_len ({cartridge_len}); "
                  f"padding by cycling to {cartridge_len}")
            sel = torch.arange(cartridge_len, device=device) % passage_len
        else:
            # Long passage: seed from cartridge_len tokens evenly strided across the
            # whole passage so tail facts are represented. The strided keys carry the
            # RoPE phase of their original positions; this is only an init — the KV are
            # free trainable parameters and distillation reshapes them, while query
            # position_ids (starting at cart_len) are unchanged.
            sel = torch.linspace(0, passage_len - 1, steps=cartridge_len, device=device).round().long()

        cartridge = QwenCartridge(num_layers, num_kv_heads, cartridge_len, head_dim)
        with torch.no_grad():
            for l in range(num_layers):
                # _layer_kv handles the DynamicCache API across transformers versions
                k, v = _layer_kv(past, l)    # each: (1, num_kv_heads, seq, head_dim)
                cartridge.keys.data[l] = k[0, :, sel, :]
                cartridge.values.data[l] = v[0, :, sel, :]

        # Free the full-corpus KV (held on GPU) before moving the cartridge over, so the
        # transient peak stays low on tight-memory large models.
        del out, past
        if device == "cuda":
            torch.cuda.empty_cache()

        param_mb = cartridge.keys.numel() * 2 * 2 / 1e6  # bfloat16 = 2 bytes, keys + values
        print(
            f"  Cartridge: {num_layers}L × {num_kv_heads}kv × "
            f"{cartridge_len}pos × {head_dim}d  ({param_mb:.1f} MB bf16)"
        )
        return cartridge.to(device)

    # ── Persistence ─────────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "num_layers": self.num_layers,
                "num_kv_heads": self.num_kv_heads,
                "cartridge_len": self.cartridge_len,
                "head_dim": self.head_dim,
            },
            path,
        )
        print(f"  Saved → {path}")

    @staticmethod
    def load(path: str | Path, device: str = "cpu") -> "QwenCartridge":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        c = QwenCartridge(
            num_layers=ckpt["num_layers"],
            num_kv_heads=ckpt["num_kv_heads"],
            cartridge_len=ckpt["cartridge_len"],
            head_dim=ckpt["head_dim"],
        )
        c.load_state_dict(ckpt["state_dict"])
        return c.eval().to(device)

    # ── M4: Context distillation ─────────────────────────────────────────────────

    @staticmethod
    def train_distillation(
        model,
        tokenizer,
        cartridge: "QwenCartridge",
        conversations: list[dict],
        passage_text: str,
        epochs: int = 10,
        steps_per_epoch: int = 40,
        lr: float = 5e-4,
        device: str = "cpu",
        checkpoint_epochs: set[int] | None = None,
        checkpoint_fn=None,
        freeze_sink: bool = True,
        full_conversation_kl: bool = False,
        teacher_corpus_snapshot=None,
        corpus_const_len: int = 0,
        no_reuse: bool = False,
        distractor_pool=None,
        max_distractors: int = 2,
        compose_geom: str = "concat",
        compose_shared_sink: bool = False,
    ):
        """
        SELF-STUDY context distillation: train cartridge KV so that
            model(cartridge | conversation) ≈ model(full_passage | conversation)
        Loss = KL(student || teacher) over the assistant answer-token logits.

        Each record in `conversations` is {"question": str, "answer": str}, where the
        answer is the model's own fluent response generated with the passage in context
        (see generate_selfstudy_data.py). Because the teacher force-decodes its own
        generation here, its per-token distribution is sharp — far cleaner targets than
        force-decoding a bare gold span.

        Teacher: model sees [system: passage] [user: question] → reference distribution
        Student: model sees [user: question] with CartridgeCache prefix → approx
                 position_ids offset by cartridge_len for correct RoPE alignment

        Only cartridge.keys and cartridge.values receive gradient.
        """
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        cartridge.train()
        cart_len = cartridge.cartridge_len
        checkpoint_epochs = checkpoint_epochs or set()

        # Freeze the attention sink (paper Fig 10): keep cartridge slot-0 K/V at its
        # init value by zeroing its gradient. Slot 0 is the first key the student
        # attends to, so a trainable sink causes the instability the paper reports.
        if freeze_sink:
            def _zero_sink_grad(grad):
                grad = grad.clone()
                grad[:, :, 0, :] = 0
                return grad
            cartridge.keys.register_hook(_zero_sink_grad)
            cartridge.values.register_hook(_zero_sink_grad)

        optimizer = torch.optim.Adam(cartridge.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.1
        )

        def eval_loss() -> tuple[float, int]:
            """Deterministic KL over ALL conversations (no sampling, no grad).
            Gives a readable convergence curve, unlike the noisy sampled train loss."""
            cartridge.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for conv in conversations:
                    out = QwenCartridge._distill_loss_for_conv(
                        model, tokenizer, cartridge, conv, passage_text,
                        cart_len, device, full_conversation_kl,
                        teacher_corpus_snapshot, corpus_const_len,
                    )
                    if out is not None:
                        total += out.item()
                        n += 1
            cartridge.train()
            return (total / max(n, 1)), n

        for epoch in range(epochs):
            total_loss = 0.0
            skipped = 0

            # no_reuse: one shuffled pass over every unique conversation (paper-faithful
            # one-epoch-no-reuse when convs are plentiful). Else: sampled with replacement.
            if no_reuse:
                step_convs = list(conversations)
                random.shuffle(step_convs)
            else:
                step_convs = [random.choice(conversations) for _ in range(steps_per_epoch)]

            for conv in step_convs:
                # Composition-aware sampling: per step draw K∈{0,1,2} frozen distractors
                # (K=0 kept in the mix to preserve single-cartridge fidelity) and place the
                # trainable cartridge at a random block among the K+1 positions.
                distractors, x_pos = None, 0
                if distractor_pool:
                    K = random.randint(0, max_distractors)   # includes K=0 (single-cartridge fidelity)
                    if K > 0:
                        distractors = random.sample(distractor_pool, min(K, len(distractor_pool)))
                        x_pos = random.randint(0, len(distractors))
                loss = QwenCartridge._distill_loss_for_conv(
                    model, tokenizer, cartridge, conv, passage_text,
                    cart_len, device, full_conversation_kl,
                    teacher_corpus_snapshot, corpus_const_len,
                    distractors=distractors, x_pos=x_pos, compose_geom=compose_geom,
                    compose_shared_sink=compose_shared_sink,
                )
                if loss is None:
                    skipped += 1
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cartridge.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            current_lr = scheduler.get_last_lr()[0]
            steps_done = len(step_convs) - skipped
            train_loss = total_loss / max(steps_done, 1)
            ev_loss, ev_n = eval_loss()
            print(
                f"  epoch {epoch + 1:3d}/{epochs}  "
                f"train_loss={train_loss:.4f}  eval_loss={ev_loss:.4f} (n={ev_n})  "
                f"lr={current_lr:.2e}  skipped={skipped}"
            )
            scheduler.step()

            if (epoch + 1) in checkpoint_epochs and checkpoint_fn is not None:
                cartridge.eval()
                checkpoint_fn(epoch + 1, cartridge)
                cartridge.train()

    @staticmethod
    def _distill_loss_for_conv(
        model, tokenizer, cartridge, conv, passage_text, cart_len, device,
        full_conversation_kl: bool = True,
        teacher_corpus_snapshot=None,
        corpus_const_len: int = 0,
        distractors=None, x_pos: int = 0, compose_geom: str = "concat",
        compose_shared_sink: bool = False, gate=None, gold_cartridges=None,
        max_answer_tokens: int | None = None,
    ):
        """
        KL(student || teacher) for one self-study conversation. With
        full_conversation_kl=True the window is the whole shared conversation span
        (user question + assistant answer, paper Eq 3); otherwise just the answer
        tokens. Returns a scalar loss tensor (with grad if cartridge requires it),
        or None if the record is unusable.

        If `teacher_corpus_snapshot` (+ `corpus_const_len`) is given, the teacher
        forward reuses the precomputed corpus-block KV and processes only the
        per-conversation suffix — essential for large corpora (see
        build_teacher_corpus_cache).
        """
        question = conv.get("question", "")
        answer = conv.get("answer", "")
        if not question or not answer:
            return None

        teacher_prefix = _build_conv_prefix(question, passage_text, tokenizer)
        student_prefix = _build_conv_prefix(question, None, tokenizer)

        teacher_full_ids, teacher_ans_start = encode_with_answer(
            teacher_prefix, answer, tokenizer
        )
        student_full_ids, student_ans_start = encode_with_answer(
            student_prefix, answer, tokenizer
        )

        if teacher_ans_start <= 0 or student_ans_start <= 0:
            return None
        if teacher_ans_start >= len(teacher_full_ids) or student_ans_start >= len(student_full_ids):
            return None

        # Cap the supervised answer window: a long answer blows up the retained student
        # activation graph (the whole 36-layer composed forward is kept for the gate/cartridge
        # gradient). The first ~N answer tokens carry the weighting signal; drop the tail.
        if max_answer_tokens and max_answer_tokens > 0:
            teacher_full_ids = teacher_full_ids[:teacher_ans_start + max_answer_tokens]
            student_full_ids = student_full_ids[:student_ans_start + max_answer_tokens]

        if full_conversation_kl:
            # Shared span = the conversation turn (user question + assistant answer),
            # which is token-identical in teacher and student from the <|im_start|>user
            # turn onward — they differ only in the leading system/cartridge block.
            # Recover it by right-aligning the two sequences (matching tail length L).
            max_L = min(len(teacher_full_ids), len(student_full_ids))
            L = 0
            while L < max_L and teacher_full_ids[-(L + 1)] == student_full_ids[-(L + 1)]:
                L += 1
            teacher_start = len(teacher_full_ids) - L
            student_start = len(student_full_ids) - L
            # Never include slot 0 (no preceding token to predict it from).
            teacher_start = max(teacher_start, 1)
            student_start = max(student_start, 1)
        else:
            teacher_start = teacher_ans_start
            student_start = student_ans_start

        teacher_inp = torch.tensor(
            teacher_full_ids[:-1], dtype=torch.long, device=device
        ).unsqueeze(0)
        student_inp = torch.tensor(
            student_full_ids[:-1], dtype=torch.long, device=device
        ).unsqueeze(0)

        # Teacher logits over the supervised window. With a precomputed corpus-KV
        # snapshot, forward only the per-conversation suffix against the cached corpus
        # block (absolute positions match the full-encode case, so RoPE is identical).
        with torch.no_grad():
            if teacher_corpus_snapshot is not None and teacher_start - 1 >= corpus_const_len:
                suffix_ids = teacher_full_ids[corpus_const_len:-1]
                suffix_inp = torch.tensor([suffix_ids], dtype=torch.long, device=device)
                pos = torch.arange(
                    corpus_const_len, corpus_const_len + len(suffix_ids),
                    dtype=torch.long, device=device,
                ).unsqueeze(0)
                cache_t = _cache_from_snapshot(teacher_corpus_snapshot)
                t_out = model(
                    input_ids=suffix_inp, past_key_values=cache_t,
                    position_ids=pos, use_cache=True,
                )
                off = (teacher_start - 1) - corpus_const_len  # suffix logit idx for teacher_start-1
                teacher_logits = t_out.logits[0, off:, :]
            else:
                # Only compute logits for the answer window (last `keep` positions), not the
                # whole corpus+conv sequence — the LM head over the full-vocab prefix is the
                # memory hog that OOMs compose-aware training on small GPUs.
                keep = teacher_inp.shape[1] - max(teacher_start - 1, 0)
                t_out = model(input_ids=teacher_inp, use_cache=False, logits_to_keep=keep)
                teacher_logits = t_out.logits[0, -keep:, :]

        # Student: question only + cartridge prefix with RoPE-offset position_ids.
        # Composition-aware training: when distractors are given, the trainable cartridge is
        # trained WITH K frozen distractor cartridges present, so it stays findable next to a
        # neighbor. Two geometries (match the two inference mechanisms):
        #   compose_geom="concat" — distinct positions (re-RoPE), X at block x_pos, query
        #       after all blocks. Trains robustness to concat's OOD-position regime.
        #   compose_geom="gnorm"  — OVERLAPPING positions (every cartridge at 0..cart_len-1,
        #       query at cart_len, joint-softmax patch), order-invariant so x_pos is moot.
        #       Trains robustness to gnorm's two-keys-per-phase regime.
        # X keeps grad (differentiable rope/cast); distractors are requires_grad=False.
        qa_len = student_inp.shape[1]
        loop_ctx = contextlib.nullcontext()
        # Two-gold (bridge/synthesis) training: the student composes BOTH gold cartridges
        # (+ optional distractors) and the teacher reads BOTH passages — so the supervised
        # target is producible only by attending to the two golds together. Used to train the
        # gate (S1b) or the cartridges (S2) on co-attention. gnorm only (order-invariant).
        golds = list(gold_cartridges) if gold_cartridges else [cartridge]
        if (gold_cartridges or distractors) and compose_geom == "gnorm":
            cache = ComposedCartridgeCache(
                golds + list(distractors or []), device=device, dtype=model.dtype, mode="gnorm",
                shared_sink=compose_shared_sink, model=model, tokenizer=tokenizer, gate=gate)
            query_off = cart_len
            loop_ctx = composed_attention(model, cache)
        elif distractors:
            components = list(distractors)
            components.insert(x_pos, cartridge)          # trainable cartridge at block x_pos
            cache = StackedCartridgeCache(components, model, device=device, dtype=model.dtype)
            query_off = len(components) * cart_len
        else:
            cache = CartridgeCache(cartridge, device=device, dtype=model.dtype)
            query_off = cart_len
        pos_ids = torch.arange(
            query_off, query_off + qa_len, dtype=torch.long, device=device
        ).unsqueeze(0)
        keep_s = qa_len - max(student_start - 1, 0)
        with loop_ctx:
            s_out = model(
                input_ids=student_inp,
                past_key_values=cache,
                position_ids=pos_ids,
                use_cache=True,
                logits_to_keep=keep_s,
            )
        student_logits = s_out.logits[0, -keep_s:, :]

        # Align answer-token lengths (off by at most 1 due to the [:-1] shift)
        min_len = min(teacher_logits.shape[0], student_logits.shape[0])
        if min_len <= 0:
            return None
        teacher_logits = teacher_logits[:min_len]
        student_logits = student_logits[:min_len]

        return F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )

    @staticmethod
    def _distill_loss_for_pair(
        model, tokenizer, cartridge, pair, passage_text,
        cart_len, device, max_passage_tokens,
    ):
        """
        Compute KL(student || teacher) for a single QA pair.
        Returns a scalar loss tensor (with grad if cartridge.requires_grad), or
        None if the pair is unusable (missing fields / degenerate tokenization).
        """
        question = pair.get("question", "")
        raw_ans = pair.get("answers", [{}])
        if raw_ans:
            answer = raw_ans[0].get("text", pair.get("answer", ""))
            ans_char_start = raw_ans[0].get("answer_start", None)
        else:
            answer = pair.get("answer", "")
            ans_char_start = None

        if not question or not answer:
            return None

        # Crop passage around the answer if needed to control teacher token budget
        teacher_passage = passage_text
        if ans_char_start is not None:
            teacher_passage = crop_context(passage_text, ans_char_start, answer)

        teacher_passage_ids = tokenizer.encode(teacher_passage, add_special_tokens=False)
        if len(teacher_passage_ids) > max_passage_tokens:
            chars_budget = max_passage_tokens * 3  # ~3 chars/token for English
            if ans_char_start is not None:
                teacher_passage = crop_context(
                    passage_text, ans_char_start, answer,
                    window_chars=chars_budget // 2,
                )
            else:
                teacher_passage = passage_text[:chars_budget]

        student_prefix = _build_prefix(question, None, tokenizer)
        teacher_prefix = _build_prefix(question, teacher_passage, tokenizer)

        student_full_ids, student_ans_start = encode_with_answer(
            student_prefix, answer, tokenizer
        )
        teacher_full_ids, teacher_ans_start = encode_with_answer(
            teacher_prefix, answer, tokenizer
        )

        if student_ans_start <= 0 or teacher_ans_start <= 0:
            return None

        teacher_inp = torch.tensor(
            teacher_full_ids[:-1], dtype=torch.long, device=device
        ).unsqueeze(0)
        student_inp = torch.tensor(
            student_full_ids[:-1], dtype=torch.long, device=device
        ).unsqueeze(0)

        # Teacher: full passage context, frozen model, no cache
        with torch.no_grad():
            t_out = model(input_ids=teacher_inp, use_cache=False)
            teacher_pred_start = teacher_ans_start - 1
            teacher_logits = t_out.logits[0, teacher_pred_start:, :]

        # Student: QA only + cartridge prefix with RoPE-offset position_ids
        qa_len = student_inp.shape[1]
        pos_ids = torch.arange(
            cart_len, cart_len + qa_len, dtype=torch.long, device=device
        ).unsqueeze(0)
        cache = CartridgeCache(cartridge, device=device, dtype=model.dtype)
        s_out = model(
            input_ids=student_inp,
            past_key_values=cache,
            position_ids=pos_ids,
            use_cache=True,
        )
        student_pred_start = student_ans_start - 1
        student_logits = s_out.logits[0, student_pred_start:, :]

        # Align lengths (off by at most 1 due to [:-1] shift)
        min_len = min(teacher_logits.shape[0], student_logits.shape[0])
        if min_len <= 0:
            return None
        teacher_logits = teacher_logits[:min_len]
        student_logits = student_logits[:min_len]

        return F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )


# ── Smoke test (M1 + M2) ────────────────────────────────────────────────────────

def smoke_test(model_id: str = "Qwen/Qwen3-1.7B"):
    """
    Sanity check: load Qwen3, compute perplexity on a hardcoded SQuAD QA pair
    under three conditions, then init a cartridge from the passage prefix.

    Expected: teacher PPL < no-cartridge PPL (full context always helps).
    The initialized cartridge PPL may or may not improve over baseline before
    training — that is checked in M2 distillation.
    """
    device = get_device()
    model, tokenizer = load_model(model_id, device)

    passage = (
        "Tesla was born on 10 July [O.S. 28 June] 1856, in the village of Smiljan, "
        "Austrian Empire (modern-day Croatia). His father, Milutin Tesla, was an Eastern "
        "Orthodox priest. His mother, Đuka Tesla (née Mandić), had a talent for making "
        "home craft tools and mechanical appliances and reciting Serbian epic poems."
    )
    question = "When was Nikola Tesla born?"
    answer = "10 July 1856"

    print(f"\nPassage: {passage[:80]}...")
    print(f"Q: {question}")
    print(f"A: {answer}\n")

    ppl_no_cart = compute_perplexity(model, tokenizer, question, answer, device=device)
    ppl_teacher = compute_perplexity(
        model, tokenizer, question, answer, passage=passage, device=device
    )

    print(f"  no-cartridge PPL:          {ppl_no_cart:8.2f}")
    print(f"  teacher (full context) PPL: {ppl_teacher:8.2f}")

    print("\nInitializing cartridge from passage prefix (cartridge_len=64) ...")
    cartridge = QwenCartridge.init_from_passage(
        model, tokenizer, passage, cartridge_len=64, device=device
    )
    ppl_cart = compute_perplexity(
        model, tokenizer, question, answer, cartridge=cartridge, device=device
    )
    print(f"  init cartridge PPL:         {ppl_cart:8.2f}")

    print()
    if ppl_teacher < ppl_no_cart:
        print("PASS — teacher PPL < no-cartridge PPL (context helps).")
    else:
        print("WARNING — teacher PPL >= no-cartridge PPL. Check chat template formatting.")

    print("\nSmoke test complete. Run train_squad_cartridges.py to train a cartridge.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3 cartridge smoke test")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test(args.model_id)
    else:
        parser.print_help()
