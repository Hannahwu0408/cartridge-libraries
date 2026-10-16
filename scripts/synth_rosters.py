"""
synth_rosters.py — parse the synth B-article ROSTERS into structured (name, town) triples.

Single source of truth for 12-I.2: both build_synth_amulti.py (which needs cart_B's 3 masters + towns to
build the multi-subject A) and classify_compose_failures.py (which needs the decoy towns to tell a
wrong-intermediate failure from a failure-to-chain) read the rosters through here. Inline regex over the
prose is fragile (a naive one-liner caught 1 of 3 masters), so parse sentence-by-sentence and VERIFY every
roster against multihop (the E2 name must match probe_a.answer and its town must match the bridge answer).

Reads the B-article contexts straight from articles_mm.jsonl (does NOT import build_synth_mm, whose
module-level code writes files).
"""
from __future__ import annotations

import re
import json
from pathlib import Path

_TOWN = re.compile(r"born in the \w+ of ([A-Z][\w']*(?: [A-Z][\w']*)*)")
_NAME = re.compile(r"^[A-Z][\w'’.-]+(?: [A-Z][\w'’.-]+)+$")   # ≥2 Capitalized tokens = a proper name


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def parse_roster(context: str) -> list[dict]:
    """Return [{name, town}, ...] for each 'X ... was born in the <type> of <Town> ...' sentence."""
    out = []
    for sent in re.split(r"(?<=\.)\s+", context.strip()):
        if "born in the" not in sent:
            continue
        pre = sent.split(" was born", 1)[0]          # name is before ' was born'
        segs = [s.strip() for s in pre.split(",")]   # pick the proper-name segment, skipping a leading
        name = next((s for s in segs if _NAME.match(s)), segs[0])  # positional phrase ("The third, X, ...")
        m = _TOWN.search(sent)                        # town = the Capitalized run after 'of '
        if not name or not m:
            continue
        out.append({"name": name, "town": m.group(1).strip()})
    return out


def get_rosters(articles_file: str, multihop_file: str) -> dict[str, list[dict]]:
    """{e2_slug: [{name, town, is_e2}] x3}, verified against multihop (E2 name+town)."""
    arts = {a["passage_id"]: a["context"] for a in load_jsonl(articles_file)}
    rosters: dict[str, list[dict]] = {}
    for q in load_jsonl(multihop_file):
        e2_slug = q["article_slugs"][1]
        e2_name, e2_town = q["probe_a"]["answer"], q["answer"]
        masters = parse_roster(arts[e2_slug])
        assert len(masters) == 3, f"{e2_slug}: parsed {len(masters)} masters, expected 3 → {masters}"
        hit = [m for m in masters if m["name"] == e2_name]
        assert len(hit) == 1, f"{e2_slug}: E2 name {e2_name!r} not uniquely parsed → {masters}"
        assert hit[0]["town"] == e2_town, f"{e2_slug}: E2 town {hit[0]['town']!r} != multihop {e2_town!r}"
        for m in masters:
            m["is_e2"] = (m["name"] == e2_name)
        rosters[e2_slug] = masters
    return rosters


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="parse + verify synth rosters")
    ap.add_argument("--articles-file", default="data/synth/articles_mm.jsonl")
    ap.add_argument("--multihop-file", default="data/synth/multihop.jsonl")
    args = ap.parse_args()
    r = get_rosters(args.articles_file, args.multihop_file)
    print(f"OK — parsed + verified {len(r)} rosters (3 masters each, E2 name+town match multihop)")
    for slug in list(r)[:3]:
        print(f"  {slug}: " + " | ".join(
            f"{m['name']}→{m['town']}{'*' if m['is_e2'] else ''}" for m in r[slug]))
