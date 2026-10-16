"""
build_synth_desc.py — ARM (b) of the 12-I break-one-property ablation (lexical anchoring).

Question: does the composition mechanism rely on a VERBATIM cross-cartridge string match? On synth the
intermediate's full name ("Vronila Teshk") appears verbatim in BOTH A ("apprenticed to Vronila Teshk")
and B (the roster line) — a lexically-anchored bridge. Here we keep the fictional name in A but make B
refer to the same entity WITHOUT that surface form, so no shared string links the two cartridges. If
compose collapses, the bridge required a shared lexical anchor — Candidate 2. (This is the most
2Wiki-faithful arm: 2Wiki routinely refers to a bridge entity by role/descriptor across passages.)

Two modes (A + multihop are UNCHANGED in both; only B's E2 line changes):
  --mode descriptor (default, user-chosen):  replace "<E2 name>, <appositive>," with an in-article
      DESCRIPTOR subject ("The pioneer of lattice-glass") → ZERO shared name token. NOTE this removes the
      identity link entirely, so a collapse mainly confirms "a shared anchor is necessary"; it does not
      separate "needs the EXACT string" from "needs ANY coreference cue".
  --mode firstname:  replace the full name with the FIRST NAME only ("Vronila, who pioneered...") → drops
      the full-name bigram but keeps a partial, still-coreferent cue. The sharper isolation of
      exact-string-dependence; run this if descriptor-only collapses to ~0 and you want to disambiguate.

Reads  data/synth/articles_mm.jsonl + data/synth/multihop.jsonl
Writes <out-dir>/articles_mm.jsonl + <out-dir>/multihop.jsonl   (default out-dir per mode)
"""
import json
import argparse
from pathlib import Path

# E2 slug -> DESCRIPTOR SUBJECT (from the E2 line's own appositive); replaces "<name>, <appositive>,"
# up to (not including) " was born", leaving the town + tail intact.
DESC = {
    "vronila_teshk":     "The pioneer of lattice-glass",
    "oqquin_vale":       "The compiler of the first winter atlas",
    "serapha_nooq":      "The deviser of the rolling chain",
    "tremaug_ossler":    "The perfecter of the river-bronze alloy",
    "vashti_qloren":     "The master of the seven-string lute",
    "iyzabel_cthonn":    "The discoverer of the binding mordant",
    "nimue_astraq":      "The navigator who read foam-color alone",
    "corabel_yrix":      "The keeper of the whistled command-line",
    "yssora_belnokh":    "The developer of the shadow-weft",
    "melisan_thornbeck": "The finest hull-builder of her age",
    "ottilda_vrace":     "The author of the great herbal",
    "garrick_umbral":    "The deviser of the acid-wash aging",
    "perdita_loom":      "The perfecter of the hair-fine escapement",
    "hedwin_carrow":     "The keeper who saved the grey bee from blight",
    "sabetha_krund":     "The foremost punch-cutter of her age",
    "wynne_torgast":     "The master of pattern-welded steel",
    "cyprian_vell":      "The foremost authority on grafting",
    "valdis_prume":      "The reviver of the crushed-lapis blue",
    "elspeth_morrow":    "The finest lens-grinder of the province",
    "rula_meschaim":     "The maker of the famed amber wine",
}


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def rewrite_b(context: str, name: str, mode: str, slug: str) -> str:
    idx = context.find(name)
    assert idx != -1, f"{name} not found in B roster {slug}"
    if mode == "firstname":
        # drop the full-name bigram, keep the first name (partial, still coreferent)
        return context[:idx] + name.split()[0] + context[idx + len(name):]
    # descriptor: replace "<name>, <appositive>," (name .. just before ' was born') with the descriptor
    born = context.find(" was born", idx)
    assert born != -1, f"' was born' not found after {name} in {slug}"
    return context[:idx] + DESC[slug] + context[born:]


def main():
    ap = argparse.ArgumentParser(description="Arm (b): break the verbatim cross-cartridge name match in B")
    ap.add_argument("--mode", choices=["descriptor", "firstname"], default="descriptor")
    ap.add_argument("--out-dir", default=None, help="default: data/synth_desc (descriptor) / data/synth_fname (firstname)")
    ap.add_argument("--selfstudy-file", default=None,
                    help="optional CONTROL self-study; rewrite the E2 reference in B-article (E2-slug) convs "
                         "ONLY (A convs keep the name, matching the arm) → <out-dir>/selfstudy.jsonl. So "
                         "cart_B never sees the name. (Bridge convs must be REGENERATED from these articles.)")
    args = ap.parse_args()

    src = Path("data/synth")
    dst = Path(args.out_dir or ("data/synth_desc" if args.mode == "descriptor" else "data/synth_fname"))
    dst.mkdir(parents=True, exist_ok=True)

    multihop = load_jsonl(src / "multihop.jsonl")
    e2name = {q["article_slugs"][1]: q["probe_a"]["answer"] for q in multihop}
    assert set(e2name) == set(DESC), f"roster/DESC mismatch: {set(e2name) ^ set(DESC)}"

    arts = load_jsonl(src / "articles_mm.jsonl")
    n = 0
    with (dst / "articles_mm.jsonl").open("w", encoding="utf-8") as f:
        for a in arts:
            slug = a["passage_id"]
            if slug in DESC:                                    # a B (E2) roster → rewrite its E2 line
                a = {**a, "context": rewrite_b(a["context"], e2name[slug], args.mode, slug)}
                n += 1
            f.write(json.dumps(a, ensure_ascii=False) + "\n")   # A articles copied unchanged
    # multihop: the bridge query (names E1) + probe_a (cart_A holds the FULL name from A) stay UNCHANGED.
    # In firstname mode, cart_B only ever saw the FIRST name, so probe_b must ask by the first name too —
    # otherwise probe_B fails on a handle cart_B never held and mis-fires the fidelity gate (as descriptor
    # mode did, probe_B=0.00). In descriptor mode we leave probes as-is (that arm is teacher-confounded).
    out_mh = []
    for q in multihop:
        q = json.loads(json.dumps(q))
        if args.mode == "firstname":
            e2 = e2name[q["article_slugs"][1]]
            q["probe_b"]["question"] = q["probe_b"]["question"].replace(e2, e2.split()[0])
        out_mh.append(q)
    (dst / "multihop.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in out_mh), encoding="utf-8")

    if args.selfstudy_file:
        # rewrite E2 reference only in B-article (E2-slug) convs; A convs unchanged (they keep the name)
        def ref(slug, name):
            return name.split()[0] if args.mode == "firstname" else DESC[slug]
        convs, nc = load_jsonl(args.selfstudy_file), 0
        for c in convs:
            slug = c.get("passage_id")
            if slug in DESC:
                r = ref(slug, e2name[slug])
                for k in ("question", "answer"):
                    if isinstance(c.get(k), str) and e2name[slug] in c[k]:
                        c[k] = c[k].replace(e2name[slug], r); nc += 1
        (dst / "selfstudy.jsonl").write_text(
            "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in convs), encoding="utf-8")
        print(f"  rewrote E2 ref in {nc} B-conv fields → {dst / 'selfstudy.jsonl'} ({len(convs)} convs total)")

    print(f"Wrote {dst}/  (mode={args.mode}, {n} B rosters rewritten, {len(multihop)} pairs)")
    ex = next(a for a in load_jsonl(dst / "articles_mm.jsonl") if a["passage_id"] == "vronila_teshk")
    print("Example B (vronila_teshk):", ex["context"][:200])


if __name__ == "__main__":
    main()
