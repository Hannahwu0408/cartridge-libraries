"""
filter_relational_selfstudy.py — ARM (c) of the 12-I ablation (self-study data mismatch).

Question: on synth, the tiny single-fact A-articles make self-study DENSE with the exact relational
framing the bridge needs ("who was Kael's mentor / who did Kael train under"). So cart_A may be answering
the bridge's hop-1 from a memorized framing, not genuine retrieval. On 2Wiki, real Wikipedia self-study
rarely reproduces the specific relational path. Test: strip the relational framing from the A-article
(E1) self-study, so cart_A's mentor link survives ONLY via raw-passage distillation, not reinforced by
relational Qs. If compose collapses (with A + B articles + B self-study unchanged), the bridge was carried
by pre-baked relational framings — Candidate 3.

Drops A-article (E1-slug) conversations whose QUESTION uses a mentor/teacher-relation framing. B (E2)
self-study and all non-E1 articles are untouched.

VALIDITY GATE: after retraining, check probe_A. If probe_A drops vs control, we removed the FACT itself
(distillation alone didn't encode it) → confounded; regenerate A self-study with a NON-relational
paraphrase of the mentor fact instead of a hard strip.

Usage:
  python scripts/filter_relational_selfstudy.py \
      --selfstudy-file data/synth/selfstudy_ca_all.jsonl \
      --multihop-file data/synth/multihop.jsonl \
      --out-file data/synth/selfstudy_norel.jsonl
"""
import re
import json
import argparse
from pathlib import Path

# Relational-framing cues for the A→E2 (mentor/teacher) hop-1 the bridge uses.
REL = re.compile(
    r"\b(mentor|mentored|apprentic|tutored?|tutor|taught|teacher|pupil|proteg|protég"
    r"|trained (?:by|under|with)|train under|studi?ed under|study under|learned from"
    r"|student of|under whom|under whose|whose (?:guidance|tutelage)|master(?:ed)? under)\b",
    re.IGNORECASE,
)


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="Arm (c): strip relational framing from A-article self-study")
    ap.add_argument("--selfstudy-file", required=True, help="the CONTROL self-study (e.g. selfstudy_ca_all.jsonl)")
    ap.add_argument("--multihop-file", default="data/synth/multihop.jsonl")
    ap.add_argument("--out-file", required=True)
    args = ap.parse_args()

    multihop = load_jsonl(args.multihop_file)
    e1_slugs = {q["article_slugs"][0] for q in multihop}     # the A (mentee) articles

    convs = load_jsonl(args.selfstudy_file)
    kept, dropped = [], []
    for c in convs:
        if c.get("passage_id") in e1_slugs and REL.search(c.get("question", "")):
            dropped.append(c)
        else:
            kept.append(c)

    Path(args.out_file).write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in kept), encoding="utf-8")

    # per-A-article drop counts — confirm the relational density is actually removed
    from collections import Counter
    dc = Counter(c["passage_id"] for c in dropped)
    print(f"Total convs {len(convs)} → kept {len(kept)}, dropped {len(dropped)} relational A-convs "
          f"across {len(dc)}/{len(e1_slugs)} A-articles")
    print("  per-A dropped (min/median/max):",
          min(dc.values(), default=0),
          sorted(dc.values())[len(dc)//2] if dc else 0,
          max(dc.values(), default=0))
    for c in dropped[:5]:
        print(f"  DROP [{c['passage_id']}] {c['question'][:90]}")
    print(f"Wrote {args.out_file}")


if __name__ == "__main__":
    main()
