"""eval_nhop.py — how does the single-cartridge CHAIN behave as hop-depth grows? (the combinatorial question)

Runs the iterative N-hop chain on the fictional chain set and reports, by depth (2/3/4):
  (A) PER-HOP DECOMPOSITION — per-hop success measured INDEPENDENTLY (start each hop from the GOLD entity):
      extraction-ok (named the correct mentor) & retrieval-ok (retrieved the correct next article). Then
      end-to-end (following the ACTUAL path) ≈ Π(per-hop) — the geometric/combinatorial decay.
  (B) CARTRIDGE-vs-TEXT — the same questions answered from the FULL TEXT of all chain articles (teacher). Does
      the chain degrade FASTER than reading everything at once, and by how much per added hop?

Reuses generate_answer (route-to-single per hop), the index embeddings for retrieval, containment for scoring.

Usage:
  python scripts/eval_nhop.py --index data/synth_chain/index.json --articles-file data/synth_chain/articles.jsonl \
      --multihop-file data/synth_chain/multihop.jsonl --cartridge-dir cartridges/synth_chain --cartridge-len 128
"""
from __future__ import annotations

import re
import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cartridge_qwen import (load_model, QwenCartridge, generate_answer, get_device,  # noqa: E402
                            _build_conv_prefix)
from evaluate_squad import containment, normalize_text                             # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def name_match(pred, gold_title):
    """Did the extraction name the gold entity? (share the distinctive surname/first token)."""
    pt, gt = set(normalize_text(pred).split()), set(normalize_text(gold_title).split())
    return len(pt & gt) >= 1 and bool(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--articles-file", required=True)
    ap.add_argument("--multihop-file", required=True)
    ap.add_argument("--cartridge-dir", required=True)
    ap.add_argument("--cartridge-len", type=int, default=128)
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--gen-max-tokens", type=int, default=48)
    ap.add_argument("--results-dir", default="results/nhop")
    a = ap.parse_args()

    idx = json.loads(Path(a.index).read_text())
    entries = idx.get("cartridges") or idx.get("entries")
    slugs = [e.get("id", e.get("passage_id")) for e in entries]
    E = np.array([e["embedding"] for e in entries], dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(idx["embedder"]); qi = idx.get("query_instruction", "")
    def embed(t):
        v = embedder.encode(qi + t, normalize_embeddings=True).astype(np.float32); return v / (np.linalg.norm(v) + 1e-12)
    def retrieve(v, exclude=()):
        for j in np.argsort(-(E @ v)):
            if slugs[j] not in exclude:
                return slugs[j]
        return None

    arts = {r["passage_id"]: r for r in load_jsonl(a.articles_file)}
    title = {s: arts[s].get("title", s) for s in arts}
    text = {s: arts[s]["context"] for s in arts}
    mh = load_jsonl(a.multihop_file)

    device = get_device()
    model, tok = load_model(a.model_id, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    cache = {}
    def load(s):
        if s not in cache:
            p = Path(a.cartridge_dir) / s / f"cartridge_{a.cartridge_len}.pt"
            cache[s] = QwenCartridge.load(p, device=device) if p.exists() else None
        return cache[s]
    # COST ACCOUNTING for the §5.6d comparison against the LangGraph agent. Only END-TO-END chain calls
    # are counted (tally["on"] is set by run_chain and is None elsewhere), so the per-hop diagnostic
    # probes in block (A) — which an agent never pays for — cannot inflate the number.
    # Same convention as §5.6e: `input tokens` = the prompt actually sent to the reader; a cartridge
    # contributes ZERO input tokens and is reported separately as KV positions.
    tally = {"on": None}

    def gen(q, cart=None, passage=None):
        if tally["on"] is not None:
            prefix = _build_conv_prefix(q, passage, tok)
            tally["on"][0] += len(tok.encode(prefix, add_special_tokens=False))
            if cart is not None:
                tally["on"][1] += a.cartridge_len
        return generate_answer(model, tok, q, cartridge=cart, passage=passage, device=device,
                               max_new_tokens=a.gen_max_tokens, conversational=True) or ""

    def extract_mentor(entity_title, cart=None, passage=None):
        q = (f"Who was the master/mentor that {entity_title} was apprenticed to? "
             f"Reply with only that person's name.")
        if cart is None and passage is None:
            return ""
        return (gen(q, cart=cart, passage=passage) or "").strip().splitlines()[0].strip()

    def run_chain(q, d, use_text):
        """Follow the ACTUAL path from E0 for d hops, answer from the final source reached. use_text → read
        article TEXT (IRCoT-style text-chain); else load the cartridge (route-to-single cartridge chain)."""
        src = lambda s: dict(passage=text[s]) if use_text else dict(cart=load(s))
        cur = retrieve(embed(q["question"]))                 # hop-0: subject article
        visited = {cur}
        for _ in range(d):
            nm = extract_mentor(title.get(cur, cur), **src(cur))
            nxt = retrieve(embed(nm), exclude=visited) if nm.strip() else None
            if nxt is None:
                break
            visited.add(nxt); cur = nxt
        fa = gen(f"In what town was {title.get(cur, cur)} born? Reply with only the town name.", **src(cur))
        return containment(fa, q["answer"])

    def run_chain_costed(q, d, use_text):
        """run_chain + the input-token / KV-position bill for THIS query."""
        tally["on"] = [0, 0]
        try:
            ok = run_chain(q, d, use_text)
        finally:
            bill = tally["on"]
            tally["on"] = None
        return ok, bill

    from collections import defaultdict
    # per-depth accumulators
    hop_extract = defaultdict(lambda: defaultdict(list))  # depth -> hop_i -> [0/1]
    hop_retr = defaultdict(lambda: defaultdict(list))
    e2e_cart = defaultdict(list)     # cartridge chain (route-to-single per hop)
    e2e_text = defaultdict(list)     # text chain (IRCoT — same decomposition, reads text) = the FAIR baseline
    cost_cart = defaultdict(list)    # per query: [input_tokens, kv_positions] for the cartridge chain
    cost_text = defaultdict(list)    # ditto for the text chain
    oneshot = defaultdict(list)      # all chain articles in context + the nested question = one-shot reasoning

    for q in mh:
        d = q["depth"]; chain = q["chain_slugs"]; targets = q["hop_targets"]  # E1..E_N slugs

        # (A) PER-HOP INDEPENDENT: each hop from the GOLD current entity (cartridge route-to-single)
        for i in range(d):
            cur = chain[i]                                   # gold current entity E_i
            nxt_name = extract_mentor(title[cur], cart=load(cur))
            ext_ok = name_match(nxt_name, title[targets[i]])
            ret = retrieve(embed(nxt_name), exclude={cur}) if nxt_name.strip() else None
            hop_extract[d][i].append(1.0 if ext_ok else 0.0)
            hop_retr[d][i].append(1.0 if ret == targets[i] else 0.0)

        # (B) END-TO-END: cartridge chain vs text chain (both decompose) vs one-shot full-text
        ok_c, bill_c = run_chain_costed(q, d, use_text=False)
        e2e_cart[d].append(ok_c)
        cost_cart[d].append(bill_c)
        ok_t, bill_t = run_chain_costed(q, d, use_text=True)
        e2e_text[d].append(ok_t)
        cost_text[d].append(bill_t)
        ctx = "\n\n".join(text[s] for s in chain)
        oneshot[d].append(containment(gen(q["question"], passage=ctx), q["answer"]))

    def m(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    Path(a.results_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n===== N-HOP CHAIN  ({len(mh)} chains · {a.model_id}) =====")
    print(f"{'depth':>5} | per-hop (extract·retrieve=combined)          | Πper-hop | cart-chain | text-chain | 1shot-fulltext")
    out = {}
    for d in sorted(e2e_cart):
        prod, parts, combs = 1.0, [], []
        for i in range(d):
            ex, rt = m(hop_extract[d][i]), m(hop_retr[d][i])
            comb = m([e * r for e, r in zip(hop_extract[d][i], hop_retr[d][i])])
            prod *= comb; combs.append(round(comb, 3))
            parts.append(f"h{i}:{ex:.2f}·{rt:.2f}={comb:.2f}")
        ec, et, os_ = m(e2e_cart[d]), m(e2e_text[d]), m(oneshot[d])
        tc = m([c[0] for c in cost_cart[d]]); kv = m([c[1] for c in cost_cart[d]])
        tt = m([c[0] for c in cost_text[d]])
        print(f"{d:>5} | {'  '.join(parts):44s} | {prod:6.2f}   | {ec:9.2f}  | {et:9.2f}  | {os_:12.2f}")
        out[d] = {"n": len(e2e_cart[d]), "per_hop_combined": combs, "prod_per_hop": round(prod, 3),
                  "cart_chain": round(ec, 3), "text_chain": round(et, 3), "oneshot_fulltext": round(os_, 3),
                  "cart_input_tokens": round(tc, 1), "cart_kv_positions": round(kv, 1),
                  "text_input_tokens": round(tt, 1)}
    print("\nRead (A) combinatorial decay: cart-chain ≈ Πper-hop ⇒ end-to-end is the PRODUCT of per-hop success;")
    print("         watch WHICH hop's combined drops (extraction vs retrieval) and how fast Πper-hop falls with depth.")
    print("Read (B) fair comparison: cart-chain vs text-chain (both DECOMPOSE) ≈ each other = cartridge ≈ text at depth;")
    print("         1shot-fulltext (all articles + nested question, no decomposition) collapses with depth ⇒ you MUST chain.")

    # (C) COST — fills the scripted-chain cell of the §5.6d agent table. Counts ONLY end-to-end chain
    # calls, so it is like-for-like with the agent's per-query input-token figure.
    print(f"\n{'depth':>5} | cart-chain input tok/q | KV pos/q | text-chain input tok/q")
    for d in sorted(out):
        print(f"{d:>5} | {out[d]['cart_input_tokens']:>21.0f} | {out[d]['cart_kv_positions']:>8.0f} "
              f"| {out[d]['text_input_tokens']:>22.0f}")
    print("Read (C): cartridge input tokens are the PROMPTS ONLY — the document costs 0 input tokens and")
    print("         appears as KV positions instead. Compare against the §5.6d agent column (3264-6548)")
    print("         to state the scripted controller's cost advantage at matched accuracy.")
    (Path(a.results_dir) / "nhop_results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {a.results_dir}/nhop_results.json")


if __name__ == "__main__":
    main()
