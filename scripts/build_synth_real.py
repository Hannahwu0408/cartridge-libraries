"""
build_synth_real.py — ARM (a) of the 12-I break-one-property ablation (parametric interference).

Question: does the composition mechanism collapse when the INTERMEDIATE (bridge) entity has a real,
high-frequency name instead of a fictional one? On synth the intermediate ("Vronila Teshk") is fully
fictional → zero parametric prior → the only signal is the cartridge. A real name ("William Douglas")
activates the pretrained population of Douglases, so the internal query the model forms into cartridge B
may be smeared across all of them. If compose collapses here (with everything else held fixed), the
2Wiki failure is parametric interference — Candidate 1.

This replaces ONLY the intermediate E2 name (in A's mentor line + B's roster line + the multihop probes),
everywhere it occurs, with a real name. Decoy masters and the ANSWER TOWN stay fictional so the answer is
still floor-0 (no parametric shortcut to the town). A per-name global string replace is complete and safe
(fictional bigrams are unique — verified: every E2 name is verbatim in both A and B).

Reads  data/synth/articles_mm.jsonl + data/synth/multihop.jsonl
Writes data/synth_real/articles_mm.jsonl + data/synth_real/multihop.jsonl
"""
import json
import argparse
from pathlib import Path

# E2 slug -> a REAL name chosen for MAXIMAL REFERENT AMBIGUITY UNDER A STRONG PRIOR — the exact 2Wiki
# failure condition ("many similar Stuarts/Douglases"), NOT mere realness. All are recurring British
# peerage/clan names with DOZENS of historical bearers across centuries, and the top homonym surnames are
# REUSED across slots (Douglas ×4, Stewart/Stuart ×3, Campbell ×2, Hamilton ×2, Gordon ×2, Graham, Bruce,
# Murray, Fraser, Scott, Lindsay) so the pool itself is dense with near-homonyms. A name with one dominant
# referent (e.g. "Emma Roberts") is deliberately AVOIDED — that is a strong but UNambiguous prior, the
# wrong variable. Gender-matched: the 4 rosters whose E2 line carries "her" (iyzabel/yssora/melisan/
# sabetha) get female names to avoid a pronoun-incoherence confound. Answer TOWNS stay fictional → floor 0.
REAL = {
    "vronila_teshk":     "William Douglas",
    "oqquin_vale":       "James Stewart",
    "serapha_nooq":      "John Campbell",
    "tremaug_ossler":    "Archibald Douglas",
    "vashti_qloren":     "George Hamilton",
    "iyzabel_cthonn":    "Mary Stuart",         # F
    "nimue_astraq":      "Robert Bruce",
    "corabel_yrix":      "Charles Murray",
    "yssora_belnokh":    "Margaret Douglas",    # F
    "melisan_thornbeck": "Elizabeth Stewart",   # F
    "ottilda_vrace":     "Alexander Gordon",
    "garrick_umbral":    "Walter Scott",
    "perdita_loom":      "Hugh Fraser",
    "hedwin_carrow":     "Andrew Stewart",
    "sabetha_krund":     "Catherine Hamilton",  # F
    "wynne_torgast":     "Colin Campbell",
    "cyprian_vell":      "Francis Gordon",
    "valdis_prume":      "Richard Graham",
    "elspeth_morrow":    "Edward Douglas",
    "rula_meschaim":     "David Lindsay",
}


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="Arm (a): give the intermediate a real, high-frequency name")
    ap.add_argument("--selfstudy-file", default=None,
                    help="optional CONTROL self-study to name-swap (question/answer) → synth_real/ "
                         "(pure swap = cleanest isolation, no regeneration variance)")
    ap.add_argument("--bridge-file", default=None,
                    help="optional CONTROL bridge convs to name-swap → synth_real/")
    args = ap.parse_args()

    src = Path("data/synth")
    dst = Path("data/synth_real"); dst.mkdir(parents=True, exist_ok=True)

    multihop = load_jsonl(src / "multihop.jsonl")
    # fictional E2 name (exact stored string) per E2 slug, from probe_a.answer
    fict = {q["article_slugs"][1]: q["probe_a"]["answer"] for q in multihop}
    assert set(fict) == set(REAL), f"roster/REAL mismatch: {set(fict) ^ set(REAL)}"
    subs = {fict[slug]: real for slug, real in REAL.items()}

    def swap(text: str) -> str:
        for a, b in subs.items():
            text = text.replace(a, b)
        return text

    arts = load_jsonl(src / "articles_mm.jsonl")
    with (dst / "articles_mm.jsonl").open("w", encoding="utf-8") as f:
        for a in arts:
            a = {**a, "context": swap(a["context"])}
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    with (dst / "multihop.jsonl").open("w", encoding="utf-8") as f:
        for q in multihop:
            q = dict(q)
            q["question"] = swap(q["question"])
            for pk in ("probe_a", "probe_b"):
                p = dict(q[pk])
                p["question"] = swap(p["question"]); p["answer"] = swap(p["answer"])
                q[pk] = p
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    def swap_convs(in_path, out_name):
        recs = load_jsonl(in_path)
        for r in recs:
            for k in ("question", "answer"):
                if isinstance(r.get(k), str):
                    r[k] = swap(r[k])
        (dst / out_name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        print(f"  name-swapped {len(recs)} convs → {dst / out_name}")

    if args.selfstudy_file:
        swap_convs(args.selfstudy_file, "selfstudy.jsonl")
    if args.bridge_file:
        swap_convs(args.bridge_file, "bridge.jsonl")

    print(f"Wrote {dst}/  ({len(arts)} articles, {len(multihop)} pairs; {len(subs)} E2 names → real)")
    print("Example:", list(subs.items())[0])


if __name__ == "__main__":
    main()
