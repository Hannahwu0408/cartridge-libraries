"""eval_style_lock.py — did terse-inclusive self-study cure the cartridge style-lock? (§5.4c)

Evaluates EXACTLY the step where the terse-decoding damage occurred: the chain's final answer,
generated from ONE cartridge given the chain's own rewritten hop-2 question (`answer_q`, recorded in
router_results). No retrieval, no index — cartridge dirs are swapped for the same questions, so the
ONLY variable is the self-study answer style the cartridge was distilled on (+ the decoding
instruction).

Grid, per question whose answering article is in the re-distilled set:
    {orig iso cart, terse cart, mixed cart} x {verbose, +SHORT}   +   {raw article text} x both

Pre-registered readout:
  orig x SHORT << orig x verbose            reproduces the lock on the subset (sanity)
  terse-cart x SHORT recovers               lock is data-borne
  terse-cart x verbose degrades             REVERSED lock -> mechanism proven
  mixed x both good                         style diversity is the fix -> Track-2 recipe
  text control ~flat across decodings       the substrate contrast, again

Usage:
  python scripts/eval_style_lock.py \
      --terse-dir cartridges/2wiki_style_terse --mixed-dir cartridges/2wiki_style_mixed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cartridge_qwen import load_model, generate_answer, get_device, QwenCartridge  # noqa: E402
from evaluate_squad import containment, exact_match, f1_score, fuzzy_containment   # noqa: E402

# MUST stay byte-identical to router.py / text_rag.py / generate_selfstudy_data.py.
SHORT = ("  Answer with ONLY the short answer itself (a few words, e.g. a name, place, or date) — "
         "no explanation, no full sentence.")


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles-file", default="data/2wiki_v2/articles.jsonl")
    ap.add_argument("--router-results", default="results/short_answer/chain_verbose/router_results.json",
                    help="VERBOSE chain run — supplies answer_q + the answering slug per question")
    ap.add_argument("--router-results-terse", default="results/short_answer/chain_short/router_results.json",
                    help="terse chain run — only used to label questions lost/kept under terse decoding")
    ap.add_argument("--slugs-file", default="data/2wiki_v2/style20_slugs.txt")
    ap.add_argument("--orig-dir", default="cartridges/2wiki_v3_iso",
                    help="the REFERENCE pool (always loaded as arm 'orig'): the 1000/pp 4B "
                         "ungrounded cartridges behind the current 0.71 chain")
    ap.add_argument("--cart-arm", action="append", metavar="NAME:DIR",
                    help="repeatable: any additional cartridge pool to evaluate alongside 'orig', "
                         "e.g. --cart-arm C:cartridges/2wiki_fid_C --cart-arm A:cartridges/2wiki_fid_A. "
                         "Replaces the old --terse-dir/--mixed-dir pair, which hardcoded the style "
                         "experiment's two arms.")
    ap.add_argument("--terse-dir", default=None, help="[deprecated] = --cart-arm terse:DIR")
    ap.add_argument("--mixed-dir", default=None, help="[deprecated] = --cart-arm mixed:DIR")
    ap.add_argument("--cartridge-len", type=int, default=128)
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--gen-max-tokens", type=int, default=48)
    ap.add_argument("--results-dir", default="results/style_lock")
    a = ap.parse_args()

    # Same latent trap as eval_generality.py: device_map="auto" shards the model across every
    # visible GPU while cartridge KV lands on cuda:0. It only worked last night because the driver
    # exported CUDA_VISIBLE_DEVICES=0.
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        sys.exit(f"ABORT: {torch.cuda.device_count()} GPUs visible — re-run pinned to one: "
                 f"CUDA_VISIBLE_DEVICES=0 python {sys.argv[0]} ...")

    slugs = [s.strip() for s in Path(a.slugs_file).read_text().splitlines() if s.strip()]
    arts = {x["passage_id"]: x["context"] for x in load_jsonl(a.articles_file)}
    rr = {r["id"]: r for r in json.loads(Path(a.router_results).read_text())["records"]}
    rt = {r["id"]: r for r in json.loads(Path(a.router_results_terse).read_text())["records"]}

    # questions answered from a re-distilled article, + their lost/kept label under terse decoding
    qs = []
    for qid, r in rr.items():
        pol = r["policies"]["chain"]
        slug = (pol.get("retrieved") or [None])[-1]
        if slug not in slugs:
            continue
        lab = ("lost" if pol["fuzzy"] == 1.0 and rt[qid]["policies"]["chain"]["fuzzy"] == 0.0
               else "kept" if pol["fuzzy"] == 1.0 else "wrong-anyway")
        qs.append({"id": qid, "slug": slug, "gold": r["answer"],
                   "q": pol.get("answer_q") or r["question"], "label": lab,
                   "recorded_verbose_pred": pol["pred"]})
    print(f"{len(qs)} questions answered from the {len(slugs)} re-distilled articles "
          f"({sum(1 for q in qs if q['label']=='lost')} lost / "
          f"{sum(1 for q in qs if q['label']=='kept')} kept / "
          f"{sum(1 for q in qs if q['label']=='wrong-anyway')} wrong-anyway)")

    device = get_device()
    model, tok = load_model(a.model_id, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    cart_dirs = {"orig": a.orig_dir}
    if a.terse_dir:
        cart_dirs["terse"] = a.terse_dir
    if a.mixed_dir:
        cart_dirs["mixed"] = a.mixed_dir
    for spec in (a.cart_arm or []):
        nm, _, d = spec.partition(":")
        if not d:
            sys.exit(f"ABORT: --cart-arm wants NAME:DIR, got {spec!r}")
        if nm in cart_dirs:
            sys.exit(f"ABORT: duplicate arm name {nm!r} (reserved: 'orig', 'text')")
        cart_dirs[nm] = d
    carts = {}   # (arm, slug) -> cartridge
    for arm, d in cart_dirs.items():
        for s in slugs:
            p = Path(d) / s / f"cartridge_{a.cartridge_len}.pt"
            if p.exists():
                carts[(arm, s)] = QwenCartridge.load(p, device=device)
        n = sum(1 for (ar, _) in carts if ar == arm)
        print(f"  arm '{arm}': loaded {n}/{len(slugs)} cartridges from {d}")
        if n == 0:
            sys.exit(f"ABORT: no cartridges for arm '{arm}' — wrong dir or training not finished?")

    def sc(pred, gold):
        return {"em": exact_match(pred, gold), "f1": f1_score(pred, gold),
                "cont": containment(pred, gold), "fuzzy": fuzzy_containment(pred, gold)}

    conds = [(arm, dec) for arm in cart_dirs for dec in ("verbose", "short")] + \
            [("text", "verbose"), ("text", "short")]
    records = []
    for i, q in enumerate(qs):
        rec = {"id": q["id"], "slug": q["slug"], "gold": q["gold"], "label": q["label"],
               "answer_q": q["q"], "preds": {}}
        for arm, dec in conds:
            question = q["q"] + (SHORT if dec == "short" else "")
            if arm == "text":
                pred = generate_answer(model, tok, question, cartridge=None,
                                       passage=arts[q["slug"]], device=device,
                                       max_new_tokens=a.gen_max_tokens, conversational=True)
            else:
                c = carts.get((arm, q["slug"]))
                if c is None:
                    continue
                pred = generate_answer(model, tok, question, cartridge=c, passage=None,
                                       device=device, max_new_tokens=a.gen_max_tokens,
                                       conversational=True)
            rec["preds"][f"{arm}/{dec}"] = {"pred": pred, **sc(pred or "", q["gold"])}
        records.append(rec)
        print(f"[{i+1}/{len(qs)}] {q['id']} ({q['label']})  " + "  ".join(
            f"{k}:f={v['fuzzy']:.0f}" for k, v in rec["preds"].items()))

    # ── summary ───────────────────────────────────────────────────────────────
    def table(subset, name):
        if not subset:
            return
        print(f"\n=== {name} (n={len(subset)}) — EM / F1 / cEM / fuzzy ===")
        print(f"  {'condition':<16} {'EM':>6} {'F1':>6} {'cEM':>6} {'fuzzy':>6}")
        out = {}
        for arm, dec in conds:
            key = f"{arm}/{dec}"
            rs = [r["preds"][key] for r in subset if key in r["preds"]]
            if not rs:
                continue
            m = {k: sum(x[k] for x in rs) / len(rs) for k in ("em", "f1", "cont", "fuzzy")}
            out[key] = m
            print(f"  {key:<16} {m['em']:>6.2f} {m['f1']:>6.2f} {m['cont']:>6.2f} {m['fuzzy']:>6.2f}")
        return out

    all_t = table(records, "ALL questions from re-distilled articles")
    table([r for r in records if r["label"] == "lost"], "LOST subset (verbose✓ → terse✗)")
    table([r for r in records if r["label"] == "kept"], "KEPT subset (correct under both)")

    # sanity: regenerated orig/verbose should reproduce the recorded chain predictions
    same = sum(1 for r, q in zip(records, qs)
               if r["preds"].get("orig/verbose", {}).get("pred") == q["recorded_verbose_pred"])
    print(f"\nsanity: orig/verbose reproduces the recorded chain pred on {same}/{len(qs)} "
          f"(greedy decode ⇒ expect ~all; large mismatch = wrong cart dir or len)")

    Path(a.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.results_dir) / "style_lock_results.json").write_text(
        json.dumps({"n": len(qs), "slugs": slugs, "summary": all_t, "records": records}, indent=2))
    print(f"wrote {a.results_dir}/style_lock_results.json")


if __name__ == "__main__":
    main()
