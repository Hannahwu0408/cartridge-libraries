"""
build_synth_amulti.py — 12-I.2 multi-subject-A cells of the {fic,real}×{single,multi-A} 2×2.

Turns article A from a SINGLE clean mentor link into a 3-row roster PARALLEL to B: it lists three mentees
apprenticed to three masters, where the three masters ARE cart_B's three roster masters. So resolving "the
mentor of <E1>" now requires SELECTION-within-A (pick E1's row among three), and a wrong resolution lands on
a DECOY master whose birthplace IS present in cart_B (a real wrong target → the classifier can tell
wrong-intermediate from failure-to-chain). E1→E2 is preserved, so the bridge answer is unchanged.

Design cautions (baked in): A is built to ~100 words = the existing single-A length, so control/arm-a stay
length-matched single controls (no dilution-vs-capacity confound). A never contains any birthplace (towns
live only in B), so the bridge still needs B.

Modes:
  fic            — masters keep their fictional names (B unchanged). Competitors present, NO parametric
                   confusion. Isolates dilution alone.
  real_homonym   — all three masters share a dense surname (William/Archibald/Edward Douglas); E2 = REAL[slug]
                   from build_synth_real. Max mutual confusability ("many Douglases").
  real_distinct  — three distinct dense-ambiguous names (William Douglas / James Stewart / John Campbell);
                   E2 = REAL[slug]. Dense priors, not homonymous — isolates the surname-collision effect.
Real modes rebuild B with the real names (towns kept fictional → floor 0) and update the multihop probes.

Writes data/synth_amulti_<mode>/{articles_mm,multihop}.jsonl
"""
from __future__ import annotations

import re
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_rosters import get_rosters, load_jsonl          # noqa: E402
from build_synth_real import REAL                          # noqa: E402  (module-level dict; import is safe)

# two fictional mentees per roster (the OTHER apprentices, so E1's row must be selected among three)
MENTEE_POOL = [
    "Dovin Marle", "Sella Prewitt", "Corun Hale", "Ivo Brant", "Tamsin Roe", "Bexley Corr",
    "Ondra Vell", "Pel Marrow", "Sethra Onn", "Corvane Ide", "Wyla Prost", "Denna Culm",
    "Rafe Ondwin", "Miriel Sack", "Lorn Vasey", "Cadby Noon", "Petra Vole", "Osk Rennel",
    "Yannic Orr", "Sabra Teel", "Dell Pryoung", "Ines Mallow", "Corbin Ashe", "Vessa Loom",
    "Talin Rooke", "Perrin Oss", "Marek Vyle", "Odalie Crest", "Bram Sellow", "Cyra Nooth",
    "Halvor Dree", "Lisette Vaun", "Corr Emsley", "Nadia Prewe", "Fenn Ostler", "Ula Marsk",
    "Tavis Orrin", "Selby Vann", "Kester Dole", "Ammi Brack",
]
# given names for homonym decoys (kept off E2's given); dense-ambiguous stock
HOMONYM_GIVENS = ["William", "Archibald", "Edward", "James", "John", "George", "Charles", "Henry",
                  "Robert", "Thomas", "Alexander", "David", "Andrew", "Walter", "Hugh", "Patrick",
                  "Colin", "Francis", "Richard", "Malcolm"]
# distinct dense-ambiguous full names for the distinct-decoy pool (varied surnames)
DISTINCT_POOL = ["James Stewart", "John Campbell", "George Hamilton", "Robert Bruce", "Charles Murray",
                 "Walter Scott", "Hugh Fraser", "Richard Graham", "David Lindsay", "Alexander Gordon",
                 "Thomas Wright", "Andrew Bell", "Francis Moore", "Henry Clarke", "Colin Ross"]

FILLER = [
    "Each pairing was entered in the guild ledger and sealed by the warden of the quarter.",
    "The apprenticeships each ran for several years, as was the custom of that generation.",
    "No master took more than a handful of students in any single season.",
    "The rolls set down the workshops and the terms of service in careful detail.",
]


# E1's name = the first Capitalized bigram+ in probe_a.question, skipping the leading "Who". Robust to ALL
# relation phrasings ("Who was the mentor of X?", "Who trained X?", "Who was X a pupil of?") — the earlier
# "of"-based regex corrupted 18/20 rows because most phrasings have no "of".
_NAME = re.compile(r"[A-Z][\w'’.-]+(?: [A-Z][\w'’.-]+)+")


def e1_name(probe_a_q: str) -> str:
    for m in _NAME.finditer(probe_a_q):
        if m.group(0) != "Who":
            return m.group(0)
    raise ValueError(f"no E1 proper name in probe_a: {probe_a_q!r}")


def build_A(rows: list[tuple[str, str]], target=100) -> str:
    """rows = [(mentee, master)] x3, first is (E1, E2). ~target words, no birthplaces."""
    body = "The guild rolls of that generation record several apprenticeships."
    for mentee, master in rows:
        body += (f" {mentee} was apprenticed to the master artisan {master}, and served a long term "
                 f"in the workshop before working alone.")
    fi = 0
    while len(body.split()) < target - 5 and fi < len(FILLER):
        body += " " + FILLER[fi]; fi += 1
    return body


def real_names(mode: str, slug: str, masters: list[dict]) -> dict[str, str]:
    """fictional master name -> display name, per roster."""
    e2_fic = next(m["name"] for m in masters if m["is_e2"])
    if mode == "fic":
        return {m["name"]: m["name"] for m in masters}
    e2_real = REAL[slug]                              # e.g. "William Douglas"
    given, surname = e2_real.split()[0], e2_real.split()[-1]
    decoys = [m["name"] for m in masters if not m["is_e2"]]
    out = {e2_fic: e2_real}
    if mode == "real_homonym":
        givens = [g for g in HOMONYM_GIVENS if g != given][:2]
        for d, g in zip(decoys, givens):
            out[d] = f"{g} {surname}"
    elif mode == "real_distinct":
        pool = [n for n in DISTINCT_POOL if n.split()[-1] != surname][:2]
        for d, n in zip(decoys, pool):
            out[d] = n
    else:
        raise ValueError(mode)
    return out


def main():
    ap = argparse.ArgumentParser(description="12-I.2 multi-subject-A cells")
    ap.add_argument("--mode", choices=["fic", "real_homonym", "real_distinct"], required=True)
    ap.add_argument("--src", default="data/synth")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    src = Path(args.src)
    dst = Path(args.out_dir or f"data/synth_amulti_{args.mode}"); dst.mkdir(parents=True, exist_ok=True)

    rosters = get_rosters(str(src / "articles_mm.jsonl"), str(src / "multihop.jsonl"))
    multihop = load_jsonl(src / "multihop.jsonl")
    arts = {a["passage_id"]: a for a in load_jsonl(src / "articles_mm.jsonl")}

    out_arts, out_mh = {}, []
    for i, q in enumerate(multihop):
        e1_slug, e2_slug = q["article_slugs"]
        masters = rosters[e2_slug]
        nm = real_names(args.mode, e2_slug, masters)          # fic name -> display name
        e1 = e1_name(q["probe_a"]["question"])
        e2_disp = nm[next(m["name"] for m in masters if m["is_e2"])]
        decoys = [m for m in masters if not m["is_e2"]]
        mentees = MENTEE_POOL[2 * i: 2 * i + 2]
        rows = [(e1, e2_disp)] + [(mentees[k], nm[decoys[k]["name"]]) for k in range(2)]

        # A: multi-subject roster (passage_id = E1 slug)
        A = {**arts[e1_slug], "context": build_A(rows)}
        out_arts[e1_slug] = A

        # B: rename masters (real modes) keeping towns; fic mode unchanged
        b_ctx = arts[e2_slug]["context"]
        for fic, disp in nm.items():
            if fic != disp:
                b_ctx = b_ctx.replace(fic, disp)
        out_arts[e2_slug] = {**arts[e2_slug], "context": b_ctx}

        # multihop: rename E2 in probes (real modes); bridge Q (names E1) + answer town unchanged
        q = json.loads(json.dumps(q))
        if args.mode != "fic":
            e2_fic = next(m["name"] for m in masters if m["is_e2"])
            q["probe_a"]["answer"] = q["probe_a"]["answer"].replace(e2_fic, e2_disp)
            q["probe_b"]["question"] = q["probe_b"]["question"].replace(e2_fic, e2_disp)
        out_mh.append(q)

    with (dst / "articles_mm.jsonl").open("w", encoding="utf-8") as f:
        for a in out_arts.values():
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    (dst / "multihop.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in out_mh), encoding="utf-8")

    ex_e1, ex_e2 = multihop[0]["article_slugs"]
    print(f"Wrote {dst}/  (mode={args.mode}, {len(out_arts)} articles, {len(out_mh)} pairs)")
    print("  A words:", len(out_arts[ex_e1]["context"].split()),
          "| A:", out_arts[ex_e1]["context"][:180])
    print("  B:", out_arts[ex_e2]["context"][:150])


if __name__ == "__main__":
    main()
