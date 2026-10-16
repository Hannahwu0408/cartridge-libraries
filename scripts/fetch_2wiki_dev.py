"""fetch_2wiki_dev.py — pull the 2WikiMultihopQA dev split from HuggingFace and dump it in a form
prepare_2wiki.py already parses (its _maybe_json/norm_context handle HF's columnar context). Verifies
one example round-trips through prepare_2wiki's normalizers before you trust it. Writes data/2wiki_raw/dev.jsonl.

Usage:  python scripts/fetch_2wiki_dev.py
"""
import sys
import json
from pathlib import Path
from collections import Counter

CANDIDATES = ["xanhho/2WikiMultihopQA", "voidful/2WikiMultihopQA", "adhityamw11/2WikiMultihopQA"]

try:
    from datasets import load_dataset
except ImportError:
    print("datasets not installed → run:  pip install datasets  (then re-run this script)")
    sys.exit(2)

out = Path("data/2wiki_raw")
out.mkdir(parents=True, exist_ok=True)
dev = out / "dev.jsonl"

sys.path.insert(0, "scripts")
from prepare_2wiki import norm_context, norm_support_titles  # noqa: E402

for ds_id in CANDIDATES:
    for split in ["validation", "dev", "test"]:
        try:
            ds = load_dataset(ds_id, split=split, trust_remote_code=True)
        except Exception as e:
            print(f"  {ds_id}:{split} → {type(e).__name__}: {str(e)[:90]}")
            continue
        ds.to_json(str(dev))
        rows = [json.loads(l) for l in dev.read_text(encoding="utf-8").splitlines() if l.strip()]
        types = Counter(r.get("type") for r in rows)
        print(f"\nOK: loaded {ds_id}:{split} → {len(rows)} examples → {dev}")
        print(f"  types: {dict(types)}")
        print(f"  keys : {list(rows[0].keys())}")
        # verify prepare_2wiki's parsers accept it
        try:
            c = norm_context(rows[0].get("context"))
            sf = norm_support_titles(rows[0].get("supporting_facts"))
            print(f"  PARSE OK — {len(c)} context paras, supporting titles {sf[:2]}")
            if not c or not sf:
                print("  WARNING: empty parse — schema may differ; inspect rows[0] before proceeding.")
            sys.exit(0)
        except Exception as e:
            print(f"  PARSE FAILED ({type(e).__name__}: {e}); trying next candidate.")
            continue

print("\nFAILED: no candidate dataset/split loaded+parsed. Report the errors above; "
      "we'll pin the correct HF id or download the official release.")
sys.exit(1)
