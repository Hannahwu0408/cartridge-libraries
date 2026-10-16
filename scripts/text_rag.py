"""text_rag.py — Text-RAG baseline on the 2Wiki set: SAME retriever (bge index) + SAME reader (Qwen3-4B),
but prepend the retrieved article TEXT instead of loading cartridges. The apples-to-apples external
baseline for the cartridge system — isolates "cartridge compression" as the only variable.

Reports, per top-k: fuzzy / containment, retrieval recall, and mean context tokens (the cost axis). Plus an
ORACLE-text row (the 2 gold articles as text) = the teacher ceiling, and a NO-CONTEXT floor.
Compare against router.py's cartridge numbers on the same 100 questions (dense@k, chain, teacher).

Usage:
  python scripts/text_rag.py --index cartridges/2wiki_v3_base1k/index.json \
      --articles-file data/2wiki_v2/articles.jsonl --multihop-file data/2wiki_v2/multihop.jsonl \
      --top-k 1,2,3 --gen-max-tokens 48
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cartridge_qwen import load_model, generate_answer, get_device      # noqa: E402
from evaluate_squad import fuzzy_containment, containment, f1_score, exact_match  # noqa: E402

# Terse-decoding instruction for --short-answer: makes the reader emit a span, so token-F1 / strict-EM are fair
# (verbose "X was born in Y." tanks F1 precision). contains-EM is robust to verbosity either way.
SHORT = ("  Answer with ONLY the short answer itself (a few words, e.g. a name, place, or date) — "
         "no explanation, no full sentence.")


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="index.json (same one router uses)")
    ap.add_argument("--articles-file", required=True)
    ap.add_argument("--multihop-file", required=True)
    ap.add_argument("--top-k", default="1,2,3")
    ap.add_argument("--modes", default="dense,chain,oracle,noctx",
                    help="comma-sep: dense (top-k sweep), chain (iterative-text-RAG = router chain over TEXT), "
                         "oracle (2 gold articles = teacher ceiling), noctx (floor)")
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--gen-max-tokens", type=int, default=48)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--short-answer", action="store_true",
                    help="append a terse-span instruction to FINAL answers so token-F1 / strict-EM are fair "
                         "(verbose sentences tank F1 precision). Needed to compare F1/EM with published systems.")
    ap.add_argument("--results-dir", default="results/text_rag")
    a = ap.parse_args()
    ks = [int(k) for k in a.top_k.split(",")]

    def sc(pred, g):   # all field-standard answer metrics for one prediction
        return {"em": exact_match(pred, g), "f1": f1_score(pred, g),
                "cont": containment(pred, g), "fuzzy": fuzzy_containment(pred, g)}

    modes = set(m.strip() for m in a.modes.split(",") if m.strip())

    idx = json.loads(Path(a.index).read_text())
    entries = idx.get("cartridges") or idx.get("entries")
    slugs = [e.get("id", e.get("passage_id")) for e in entries]
    E = np.array([e["embedding"] for e in entries], dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(idx["embedder"])
    q_instr = idx.get("query_instruction", "")
    def embed(text):
        v = embedder.encode(q_instr + text, normalize_embeddings=True).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def retrieve(qv, k, exclude=()):
        ex = set(exclude)
        out = []
        for j in np.argsort(-(E @ qv)):
            s = slugs[j]
            if s not in ex:
                out.append(s)
                if len(out) == k:
                    break
        return out

    arts = {aa["passage_id"]: aa["context"] for aa in load_jsonl(a.articles_file)}
    mh = load_jsonl(a.multihop_file)
    if a.max_questions:
        mh = mh[: a.max_questions]

    device = get_device()
    model, tok = load_model(a.model_id, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    def ntok(text):
        return len(tok.encode(text, add_special_tokens=False))

    def ans(question, passage, short=False):
        # conversational=True matches router.py's teacher generation → oracle-text row is directly
        # comparable to the router teacher. `short` (with --short-answer) appends a terse-span instruction,
        # applied ONLY to FINAL answers (not the bridge-extract/reframe intermediates) for a fair token-F1.
        q = question + SHORT if (short and a.short_answer) else question
        return generate_answer(model, tok, q, cartridge=None, passage=passage,
                               device=device, max_new_tokens=a.gen_max_tokens, conversational=True)

    def text_of(slug_list):
        return "\n\n".join(arts[s] for s in slug_list if s in arts)

    def chain_text(q_text):
        """Iterative-text-RAG (IRCoT-style) = router.py's `chain` controller reading TEXT not cartridges.
        hop-1: retrieve top-1 by query; extract ONLY the bridge entity from that article (router's
        bridge_extract prompt). hop-2: retrieve top-1 by the bridge entity (excluding hop-1); reframe the
        multi-hop question to ask DIRECTLY about the bridge entity (router's reframe_question prompt);
        answer from the hop-2 article ALONE. The fully-fair multi-hop baseline for the cartridge chain."""
        qv = embed(q_text)
        h1 = retrieve(qv, 1)
        h1_text = text_of(h1)
        # hop-1: bridge_extract (verbatim from router.py:209-214)
        bridge_instr = (f"Question: {q_text}\n\nThis question cannot be fully answered from the context "
                        f"alone — it refers to another entity you must look up next. Identify the single "
                        f"bridge entity named in the context that links to the answer (the specific "
                        f"person, organization, or work the question is really asking about). Do NOT "
                        f"answer the question itself, and do not name a location mentioned only in "
                        f"passing. Reply with only that entity's name.")
        br = (ans(bridge_instr, h1_text) or "").strip().splitlines()
        bridge = br[0].strip() if br else ""
        ret, used_q = h1, q_text
        if bridge:
            h2 = retrieve(embed(bridge), 1, exclude=set(h1))
            if h2:
                ret = h2                                   # answer from the bridge-target article ALONE
                # reframe_question (verbatim from router.py:223-229)
                reframe_instr = (f'Rewrite the question below so it asks DIRECTLY about "{bridge}". Replace '
                                 f'the phrase describing the entity (e.g. "the father of X", "the director '
                                 f'of Y", "X\'s wife") with "{bridge}", but KEEP asking for the EXACT SAME '
                                 f'thing — the same who/what/where/when/why/how and the same attribute (e.g. '
                                 f'cause of death, date of death, birthplace, nationality, where they '
                                 f'worked/studied). Change ONLY the entity, never what is being asked.\n\n'
                                 f'Original: "{q_text}"\n\nReply with ONLY the rewritten question.')
                rq = (ans(reframe_instr, None) or "").strip().splitlines()
                used_q = (rq[0].strip().strip('"') if rq else "") or q_text
        # short=True is REQUIRED here: this is the chain's FINAL answer, and without it the IRCoT
        # baseline decodes verbosely while dense/oracle/noctx (and router.py's cartridge chain)
        # decode tersely — which silently flatters whichever system got the terse instruction.
        # Symptom of the bug: iter-chain EM 0.000 / F1 0.168 beside dense EM 0.300 / F1 0.412.
        pred = ans(used_q, text_of(ret), short=True)
        return pred, ret

    agg = {k: {"scores": [], "recall": [], "ctx": []} for k in ks}
    oracle_sc, oracle_ct, noctx_sc = [], [], []
    chain_sc, chain_ct, chain_rec = [], [], []
    # per-question predictions, so these baselines can be blind-audited alongside the cartridge
    # chain (aggregates alone cannot be re-scored, and lexical metrics disagree with meaning).
    records = {}
    for i, q in enumerate(mh):
        gold = set(q.get("article_slugs", []))
        rec = records.setdefault(q.get("id", i), {"id": q.get("id", i),
                                                 "question": q["question"], "gold": q["answer"]})
        if "dense" in modes:
            qv = embed(q["question"])
            order = np.argsort(-(E @ qv))                   # cosine (both normalized)
            for k in ks:
                topk = [slugs[j] for j in order[:k]]
                text = text_of(topk)
                pred = ans(q["question"], text, short=True)
                agg[k]["scores"].append(sc(pred, q["answer"]))
                agg[k]["recall"].append(len(gold & set(topk)) / max(1, len(gold)))
                agg[k]["ctx"].append(ntok(text))
                rec[f"dense@{k}"] = pred
        if "chain" in modes:
            cp, cret = chain_text(q["question"])
            chain_sc.append(sc(cp, q["answer"]))
            chain_ct.append(ntok(text_of(cret)))
            chain_rec.append(len(gold & set(cret)) / max(1, len(gold)))
            rec["iter_chain"] = cp
        if "oracle" in modes:
            gtext = text_of(q.get("article_slugs", []))
            op = ans(q["question"], gtext, short=True)
            oracle_sc.append(sc(op, q["answer"])); oracle_ct.append(ntok(gtext))
            rec["oracle_text"] = op
        if "noctx" in modes:
            np_ = ans(q["question"], None, short=True)
            noctx_sc.append(sc(np_, q["answer"])); rec["no_context"] = np_
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(mh)}]")

    def m(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def agm(scores):   # mean of each metric over a list of score dicts
        return {kk: m([s[kk] for s in scores]) for kk in ("em", "f1", "cont", "fuzzy")} if scores else {}

    Path(a.results_dir).mkdir(parents=True, exist_ok=True)
    tag = "SHORT-ANSWER" if a.short_answer else "verbose"
    print(f"\n===== TEXT-RAG  ({len(mh)} Qs · {a.model_id} · {tag}) — EM / F1 / contains-EM =====")
    print(f"  {'policy':16s} {'EM':>6} {'F1':>6} {'cont-EM':>8} {'fuzzy':>6} {'recall':>7} {'ctx_tok':>8}")

    def line(label, s, recall="—", ctx="—"):
        print(f"  {label:16s} {s.get('em',0):6.3f} {s.get('f1',0):6.3f} {s.get('cont',0):8.3f} "
              f"{s.get('fuzzy',0):6.3f} {recall:>7} {ctx:>8}")
    if "noctx" in modes:
        line("no-context", agm(noctx_sc))
    for k in ks:
        if "dense" in modes:
            d = agg[k]
            line(f"text-RAG@{k}", agm(d["scores"]), f"{m(d['recall']):.2f}", f"{m(d['ctx']):.0f}")
    if "chain" in modes:
        line("iter-chain(IRCoT)", agm(chain_sc), f"{m(chain_rec):.2f}", f"{m(chain_ct):.0f}")
    if "oracle" in modes:
        line("oracle-text(2)", agm(oracle_sc), "1.00", f"{m(oracle_ct):.0f}")

    out = {"n": len(mh), "modes": sorted(modes), "short_answer": a.short_answer,
           "no_context": agm(noctx_sc) if "noctx" in modes else None,
           "oracle_text": {**agm(oracle_sc), "ctx_tok": m(oracle_ct)} if "oracle" in modes else None,
           "iter_chain": {**agm(chain_sc), "recall": m(chain_rec), "ctx_tok": m(chain_ct)} if "chain" in modes else None,
           "by_k": {k: {**agm(d["scores"]), "recall": m(d["recall"]), "ctx_tok": m(d["ctx"])}
                    for k, d in agg.items()} if "dense" in modes else None,
           "records": list(records.values())}
    (Path(a.results_dir) / "text_rag_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.results_dir}/text_rag_results.json")


if __name__ == "__main__":
    main()
