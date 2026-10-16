"""eval_generality.py — did terse-only self-study destroy the cartridge's OTHER query modes?

The style-lock re-distill (§5.4c) showed terse self-study buys terse-QA competence and costs
conversational-QA competence. But `--terse-frac` appends the span instruction to EVERY seed mode,
so the pure-terse cartridge's training data contains ZERO summarization and ZERO structuring
answers (measured: mean answer 5.0 words vs 71.5 in the conversational arm). Generality across
query types is the entire premise of the CARTRIDGES paper — their five seed prompts exist to
preserve it — so if terse training collapses those modes, "style-specialized memory" is a much
sharper claim than "worse at verbose QA", and it changes which full-pool arm is worth 2.5 days.

Probes each re-distilled article with a SUMMARIZATION and a STRUCTURING request (no gold answer
exists for these, so we measure FORM and GROUNDING rather than correctness):
  words         mean answer length
  complies      summarization: >=25 words & >=2 sentences · structuring: JSON braces or a >=2-item list
  grounded      >=half the answer's content words appear in the source article (hallucination proxy,
                same rule as check_selfstudy_quality.py:81)
  coverage      (summarization) share of the article's salient terms (capitalised names / numbers)
                that appear in the summary
Raw TEXT is the ceiling (same frozen model reading the article); `orig` is the pre-intervention
cartridge. ~200 generations, 12-15 min on one GPU.

Usage:
  python scripts/eval_generality.py --terse-dir cartridges/2wiki_style_terse \
      --mixed-dir cartridges/2wiki_style_mixed
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cartridge_qwen import load_model, generate_answer, get_device, QwenCartridge  # noqa: E402

STOP = {"the", "and", "that", "with", "from", "this", "were", "which", "their", "there", "have",
        "been", "also", "into", "when", "where", "what", "they", "them", "then", "than", "some",
        "such", "these", "those", "about", "after", "before", "other", "would", "could", "being"}


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def content_words(s):
    return [w for w in re.findall(r"[a-z]{4,}|\d{2,}", (s or "").lower()) if w not in STOP]


def grounded(ans, ctx):
    w = content_words(ans)
    if not w:
        return 0.0                      # an empty/functionless answer is not grounded here
    c = ctx.lower()
    return float(sum(x in c for x in w) >= 0.5 * len(w))


def salient_terms(ctx, k=12):
    """capitalised multi-char names + numbers — the things a faithful summary should retain"""
    terms = re.findall(r"\b[A-Z][a-z]{3,}\b|\b\d{3,4}\b", ctx)
    seen, out = set(), []
    for t in terms:
        if t.lower() not in seen and t.lower() not in STOP:
            seen.add(t.lower())
            out.append(t)
    return out[:k]


def complies(mode, ans):
    a = (ans or "").strip()
    if mode == "summarization":
        # 20 words + 2 sentences separates a SUMMARY from a SPAN (terse answers measured at
        # 5.0 words, conversational self-study at 71.5) without rejecting a terse-but-real
        # 3-sentence summary. The text arm's compliance is the threshold's own sanity check —
        # main() warns if it drops below 0.8, which would mean the bar is set wrong, not that
        # the cartridges failed.
        return float(len(a.split()) >= 20 and len(re.findall(r"[.!?]", a)) >= 2)
    # structuring: JSON object/array, or a list of >=2 bullet/numbered/`key:` lines
    if a.startswith("```") and len(a.split()) <= 2:
        return 0.0                      # bare code fence = truncated by the newline stop, not JSON
    if re.search(r"[{\[].*[}\]]", a, re.S):
        return 1.0
    # single-line fallback: >=2 "key": value pairs even if the braces were cut off
    if len(re.findall(r'"[\w ]+"\s*:', a)) >= 2:
        return 1.0
    lines = [l.strip() for l in a.splitlines() if l.strip()]
    listish = sum(1 for l in lines if re.match(r"^([-*•]|\d+[.)]|[A-Za-z][\w ]{0,30}:)", l))
    return float(listish >= 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles-file", default="data/2wiki_v2/articles.jsonl")
    ap.add_argument("--slugs-file", default="data/2wiki_v2/style20_slugs.txt")
    ap.add_argument("--orig-dir", default="cartridges/2wiki_v3_iso")
    ap.add_argument("--terse-dir", default=None)
    ap.add_argument("--mixed-dir", default=None)
    ap.add_argument("--cartridge-len", type=int, default=128)
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--gen-max-tokens", type=int, default=160)
    ap.add_argument("--results-dir", default="results/style_lock")
    a = ap.parse_args()

    # load_model uses device_map="auto": with >1 visible GPU it SHARDS the model across them, while
    # cartridge KV loads onto cuda:0 -> "Expected all tensors to be on the same device" deep inside
    # the attention cat. Every driver in this repo exports CUDA_VISIBLE_DEVICES; hand-run commands
    # must too. Fail loudly here rather than 200 lines into a traceback.
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        sys.exit(f"ABORT: {torch.cuda.device_count()} GPUs visible — device_map='auto' would shard "
                 f"the model while cartridges load on cuda:0.\n"
                 f"       Re-run pinned to one GPU:  CUDA_VISIBLE_DEVICES=0 python {sys.argv[0]} ...")

    slugs = [s.strip() for s in Path(a.slugs_file).read_text().splitlines() if s.strip()]
    arts = {x["passage_id"]: x for x in load_jsonl(a.articles_file)}
    slugs = [s for s in slugs if s in arts]
    print(f"probing {len(slugs)} re-distilled articles × 2 modes × 4 arms "
          f"= {len(slugs)*2*4} generations")

    device = get_device()
    model, tok = load_model(a.model_id, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    dirs = {"orig": a.orig_dir}
    if a.terse_dir:
        dirs["terse"] = a.terse_dir
    if a.mixed_dir:
        dirs["mixed"] = a.mixed_dir
    carts = {}
    for arm, d in dirs.items():
        for s in slugs:
            p = Path(d) / s / f"cartridge_{a.cartridge_len}.pt"
            if p.exists():
                carts[(arm, s)] = QwenCartridge.load(p, device=device)
        print(f"  arm '{arm}': {sum(1 for k in carts if k[0]==arm)}/{len(slugs)} cartridges")

    arms = list(dirs) + ["text"]
    PROBES = {
        "summarization": "Summarize the key information about {title} in a short paragraph.",
        # ONE LINE, NO FENCES is not stylistic pedantry: generate_answer stops at the first
        # newline once content has begun (cartridge_qwen.py:101) and returns split("\n")[0],
        # so a fenced reply ("```json\n{...}") is truncated to the fence. Round 1 of this probe
        # scored 1.0 words / 0.00 compliance for ALL FOUR arms, text included — a harness
        # artifact, not a model property. Keeping the answer on one line measures the JSON.
        "structuring": ("List the key facts about {title} as a JSON object with descriptive "
                        "field names. Reply with the JSON on a SINGLE LINE, with no markdown "
                        "code fences and no explanation."),
    }

    records = []
    for i, s in enumerate(slugs):
        ctx, title = arts[s]["context"], arts[s].get("title", s)
        sal = salient_terms(ctx)
        for mode, tmpl in PROBES.items():
            q = tmpl.format(title=title)
            rec = {"slug": s, "mode": mode, "question": q, "preds": {}}
            for arm in arms:
                if arm == "text":
                    pred = generate_answer(model, tok, q, cartridge=None, passage=ctx,
                                           device=device, max_new_tokens=a.gen_max_tokens,
                                           conversational=True)
                else:
                    c = carts.get((arm, s))
                    if c is None:
                        continue
                    pred = generate_answer(model, tok, q, cartridge=c, passage=None,
                                           device=device, max_new_tokens=a.gen_max_tokens,
                                           conversational=True)
                pred = pred or ""
                cov = (sum(1 for t in sal if t.lower() in pred.lower()) / len(sal)) if sal else 0.0
                rec["preds"][arm] = {"pred": pred, "words": len(pred.split()),
                                     "complies": complies(mode, pred),
                                     "grounded": grounded(pred, ctx), "coverage": cov}
            records.append(rec)
        print(f"[{i+1}/{len(slugs)}] {s[:34]:<34} " + "  ".join(
            f"{arm}:{records[-1]['preds'].get(arm,{}).get('words','-'):>3}w" for arm in arms))

    print("\n" + "=" * 70)
    for mode in PROBES:
        rs = [r for r in records if r["mode"] == mode]
        print(f"\n=== {mode.upper()} probe (n={len(rs)}) ===")
        print(f"  {'arm':<8} {'words':>7} {'complies':>9} {'grounded':>9} {'coverage':>9}")
        for arm in arms:
            v = [r["preds"][arm] for r in rs if arm in r["preds"]]
            if not v:
                continue
            f = lambda k: sum(x[k] for x in v) / len(v)
            print(f"  {arm:<8} {f('words'):>7.1f} {f('complies'):>9.2f} {f('grounded'):>9.2f} "
                  f"{f('coverage'):>9.2f}")

    # metric self-check: `text` is the same frozen model reading the article, so it defines what
    # a compliant answer looks like. If IT fails the bar, the bar is wrong — not the cartridges.
    for mode in PROBES:
        v = [r["preds"]["text"] for r in records if r["mode"] == mode and "text" in r["preds"]]
        tc = sum(x["complies"] for x in v) / len(v) if v else 0.0
        if tc < 0.8:
            print(f"\n  ⚠️  METRIC WARNING: text-arm compliance on {mode} is only {tc:.2f} — the "
                  f"threshold is mis-set for this probe; read `words` and the samples instead of "
                  f"`complies`.")

    print("\n=== sample outputs (first article, both modes) ===")
    for r in records[:2]:
        print(f"\n  [{r['mode']}] {r['question']}")
        for arm in arms:
            if arm in r["preds"]:
                print(f"    {arm:<6}: {r['preds'][arm]['pred'][:150]!r}")

    Path(a.results_dir).mkdir(parents=True, exist_ok=True)
    out = Path(a.results_dir) / "generality_results.json"
    out.write_text(json.dumps({"n_articles": len(slugs), "records": records}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
