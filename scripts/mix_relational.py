"""
mix_relational.py — build the budget-matched relational-pool self-study for 12-I.4.

For A (E1) articles: relational-binding convs + base convs, capped at --cap total (so A stays at the SAME
budget as the clean-baseline pool → isolates FRAMING from training budget). For B articles: base convs
(capped). Keeps the intervention to "which framings cart_A sees," not "how much A is trained."
"""
import json
import argparse
import collections
from pathlib import Path


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--relational", required=True)
    ap.add_argument("--multihop", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=120, help="per-article conv cap (= baseline budget)")
    a = ap.parse_args()

    base, rel = load(a.base), load(a.relational)
    e1 = {q["article_slugs"][0] for q in load(a.multihop)}
    byp = collections.defaultdict(list)
    for c in base:
        byp[c["passage_id"]].append(c)
    relp = collections.defaultdict(list)
    for c in rel:
        relp[c["passage_id"]].append(c)

    out = []
    for pid, convs in byp.items():
        if pid in e1:                                    # A: relational first, fill to cap with base
            r = relp.get(pid, [])
            out += r + convs[: max(0, a.cap - len(r))]
        else:                                            # B: base only, capped
            out += convs[: a.cap]

    Path(a.out).write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in out), encoding="utf-8")
    nrel = sum(1 for c in out if c.get("seed_type") == "mentor_binding")
    print(f"wrote {len(out)} convs ({nrel} relational, cap={a.cap}) → {a.out}")


if __name__ == "__main__":
    main()
