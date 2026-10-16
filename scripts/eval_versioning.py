"""eval_versioning.py — probe an updated cartridge pool and tally whether each update TOOK (fresh, the novel
counterfactual token) or leaked STALE (the original fact). Used for Q1 (warm-start overwrite) of the versioning
experiment; reused across the {inplace, append} × {epoch} variants and for the v1 baseline.

fresh = the novel fact_v2 token appears in the answer (unambiguous — it has zero parametric prior).
stale = the original fact_v1 (or its distinctive head word) appears.
other = neither (refusal / unrelated).

Usage:
  python scripts/eval_versioning.py --cartridge-dir cartridges/2wiki_ver_inplace_e2 \
      --updates-file data/2wiki_versioned/updates.jsonl --cartridge-len 128 --label inplace_e2
  # v1 baseline (should be ~all stale) + floor:
  python scripts/eval_versioning.py --cartridge-dir cartridges/2wiki_v3_iso --updates-file ... --label v1
  python scripts/eval_versioning.py --no-cartridge --updates-file ... --label floor
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cartridge_qwen import load_model, QwenCartridge, generate_answer, get_device  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def head(fact: str) -> str:
    return fact.strip().split(",")[0].split()[0].lower() if fact.strip() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates-file", required=True)
    ap.add_argument("--cartridge-dir", default=None, help="pool to probe (omit with --no-cartridge)")
    ap.add_argument("--no-cartridge", action="store_true", help="floor: probe with no cartridge")
    ap.add_argument("--passage-dir", default=None,
                    help="TEACHER-FORCED ceiling: read <dir>/articles.jsonl and feed each edited article "
                         "TEXT in context (no cartridge). Tests whether the model reads the new fact from "
                         "EXPLICIT evidence or overrides it with its prior — the ceiling any cartridge could hit.")
    ap.add_argument("--cartridge-len", type=int, default=128)
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--gen-max-tokens", type=int, default=32)
    ap.add_argument("--label", default="run")
    ap.add_argument("--results-dir", default="results/versioning")
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()

    device = get_device()
    model, tok = load_model(a.model_id, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    ups = load_jsonl(a.updates_file)
    cd = Path(a.cartridge_dir) if a.cartridge_dir else None
    pmap = None
    if a.passage_dir:
        pmap = {j["passage_id"]: j["context"]
                for j in load_jsonl(Path(a.passage_dir) / "articles.jsonl")}

    fresh = stale = other = n = 0
    rows = []
    for u in ups:
        cart, passage = None, None
        if pmap is not None:                                  # teacher-forced: edited text in context, no cart
            passage = pmap.get(u["slug"])
            if passage is None:
                continue
        elif not a.no_cartridge:
            p = cd / u["slug"] / f"cartridge_{a.cartridge_len}.pt"
            if not p.exists():
                continue
            cart = QwenCartridge.load(p, device=device)
        ans = generate_answer(model, tok, u["probe"], cartridge=cart, passage=passage,
                              device=device, max_new_tokens=a.gen_max_tokens, conversational=True) or ""
        al = ans.lower()
        v2, v1, v1h = u["fact_v2"].lower(), u["fact_v1"].lower(), head(u["fact_v1"])
        if v2 in al:
            cls = "fresh"; fresh += 1
        elif v1 in al or (v1h and v1h in al):
            cls = "stale"; stale += 1
        else:
            cls = "other"; other += 1
        n += 1
        rows.append((cls, u["slug"], u["fact_v1"], u["fact_v2"], ans.replace("\n", " ")[:70]))

    def pct(x):
        return f"{x/n:.2f}" if n else "—"

    print(f"\n=== VERSIONING [{a.label}]  n={n} ===")
    print(f"  FRESH (update took, novel token): {fresh}  ({pct(fresh)})")
    print(f"  STALE (original fact leaked)    : {stale}  ({pct(stale)})")
    print(f"  other                           : {other}  ({pct(other)})")
    print("  --- samples ---")
    for cls, slug, v1, v2, ans in rows[: a.show]:
        print(f"   [{cls:5s}] {slug[:22]:22s} {v1!r}→{v2!r}  pred={ans!r}")

    Path(a.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.results_dir) / f"{a.label}.json").write_text(json.dumps(
        {"label": a.label, "n": n, "fresh": fresh, "stale": stale, "other": other,
         "rows": [dict(cls=c, slug=s, v1=x, v2=y, pred=p) for c, s, x, y, p in rows]}, indent=2))
    print(f"  wrote {a.results_dir}/{a.label}.json")


if __name__ == "__main__":
    main()
