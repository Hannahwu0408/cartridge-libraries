"""build_versioned_updates.py — construct COUNTERFACTUAL fact-updates on REAL 2Wiki articles for the
versioning experiment (CL-3). For each of ~N articles we take a clean proper-noun fact that appears verbatim
in the article, and replace it with a NOVEL invented token (zero parametric prior → outputting it proves the
update took; outputting the original = stale, regardless of whether that's KV residue or the model's prior).

Emits two v2 article variants per update (same passage_id as v1, so `train_joint --init-from 2wiki_v3_iso`
warm-starts each from its existing cartridge):
  inplace/  — the old fact is REPLACED by the novel token where it occurs (old fact gone) → pure overwrite test.
  append/   — the old fact stays in the body, and a corrected sentence with the novel token is APPENDED LAST
              (both present, new is last) → the recency/"last-wins" test motivated by the AB/BA ordering finding.
Plus updates.jsonl: {slug, fact_v1, fact_v2, probe} where probe is a cloze that elicits the fact.

Usage:
  python scripts/build_versioned_updates.py --articles data/2wiki_v2/articles.jsonl \
      --multihop data/2wiki_v2/multihop.jsonl --n 20 --out-dir data/2wiki_versioned
"""
from __future__ import annotations

import re
import json
import random
import argparse
from pathlib import Path

# --mode novel: clearly-invented tokens (zero prior) — the hardest case, the update fights prior + implausibility.
NOVEL = ["Zorvia", "Qelmoor", "Vandrel", "Ostrymm", "Kaeltis", "Brintaq", "Marnholt", "Vesperia",
         "Draymoor", "Elltowyn", "Sarnvale", "Thaddwick", "Ulmarra", "Crenn Holt", "Ythanor",
         "Belcarne", "Ovstrand", "Wexlyn", "Halvarr", "Nprum", "Skellow", "Tarnwick", "Vroomel",
         "Cassdrel", "Pellandor"]
# --mode real: plausible REAL replacements (a real update to a different real value) — isolates prior-conflict
# from token-implausibility. Routed by the question: place-facts → a real city; person-facts → a real name.
REAL_PLACES = ["Paris", "Vienna", "Madrid", "Berlin", "Rome", "Lisbon", "Prague", "Oslo", "Dublin",
               "Athens", "Warsaw", "Munich", "Naples", "Lyon", "Geneva", "Seville", "Krakow", "Bergen",
               "Toledo", "Bologna", "Antwerp", "Zurich", "Porto", "Ghent"]
REAL_NAMES = ["Henry of Navarre", "Eleanor of Aquitaine", "Frederick of Saxony", "Margaret of Anjou",
              "Philip of Burgundy", "Isabella of Castile", "Charles of Valois", "Beatrice of Savoy",
              "William of Orange", "Matilda of Tuscany", "Robert of Artois", "Blanche of Castile"]
PLACE_KW = {"where", "city", "country", "place", "located", "nationality", "region", "town", "born"}
PERSON_KW = {"who", "father", "mother", "wife", "husband", "son", "daughter", "spouse", "grandmother",
             "grandfather", "brother", "sister", "parent", "grandson", "granddaughter"}


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


PROPER = re.compile(r"^[A-Z][A-Za-z.'’,\- ]{2,40}$")


def is_proper_noun(s: str) -> bool:
    """Place/name-like: starts capital, ≤5 words, no digits (skip dates/numbers so the swap reads
    naturally). Allows commas/hyphens for 'City, State' and 'Polish-Lithuanian Commonwealth' forms."""
    s = s.strip()
    return (bool(PROPER.match(s)) and not any(c.isdigit() for c in s)
            and 1 <= len(s.split()) <= 5)


def sentence_with(context: str, fact: str) -> str | None:
    for sent in re.split(r"(?<=[.!?])\s+", context):
        if fact in sent:
            return sent.strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", required=True)
    ap.add_argument("--multihop", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["novel", "real"], default="novel",
                    help="novel = invented token (fights prior + implausibility); real = a different "
                         "plausible REAL value (isolates prior-conflict from implausibility)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    arts = {j["passage_id"]: j for j in load_jsonl(a.articles)}
    mh = load_jsonl(a.multihop)
    rng = random.Random(a.seed)
    novel = NOVEL[:]; rng.shuffle(novel)
    places = REAL_PLACES[:]; rng.shuffle(places)
    names = REAL_NAMES[:]; rng.shuffle(names)

    def pick_v2(question, fact):
        if a.mode == "novel":
            return novel.pop() if novel else None
        ql = (question or "").lower()
        person = any(k in ql for k in PERSON_KW) and not any(k in ql for k in PLACE_KW)
        primary, backup = (names, places) if person else (places, names)
        for pool in (primary, backup):
            while pool:
                cand = pool.pop()
                if cand.lower() != fact.lower() and cand.lower() not in fact.lower() \
                        and fact.lower() not in cand.lower():
                    return cand
        return None

    updates, used = [], set()
    for q in mh:
        if len(updates) >= a.n:
            break
        fact = (q.get("answer") or "").strip()
        if not is_proper_noun(fact):
            continue
        golds = list(q.get("article_slugs") or q.get("gold") or [])
        # the article that actually STATES the fact verbatim = the one whose cartridge we edit
        target = next((s for s in golds if s in arts and fact in arts[s]["context"]), None)
        if target is None or target in used:
            continue
        sent = sentence_with(arts[target]["context"], fact)
        if not sent:
            continue
        fact_v2 = pick_v2(q.get("question", ""), fact)
        if not fact_v2:
            continue
        used.add(target)
        cloze = "Fill in the blank with a single term: " + sent.replace(fact, "____", 1)
        updates.append({"slug": target, "title": arts[target].get("title", target),
                        "fact_v1": fact, "fact_v2": fact_v2, "probe": cloze,
                        "sentence": sent})

    if len(updates) < a.n:
        print(f"WARNING: only {len(updates)} clean proper-noun updates found (asked {a.n}).")

    out = Path(a.out_dir)
    (out / "inplace").mkdir(parents=True, exist_ok=True)
    (out / "append").mkdir(parents=True, exist_ok=True)

    def art_record(slug, context):
        base = arts[slug]
        return {"passage_id": slug, "title": base.get("title", slug), "passage_idx": 0,
                "context": context, "qas": []}

    inplace, append = [], []
    for u in updates:
        ctx = arts[u["slug"]]["context"]
        new_sent = u["sentence"].replace(u["fact_v1"], u["fact_v2"])
        inplace.append(art_record(u["slug"], ctx.replace(u["fact_v1"], u["fact_v2"])))       # old gone
        append.append(art_record(u["slug"], ctx.rstrip() + " Updated record: " + new_sent))   # old stays, new last

    (out / "inplace" / "articles.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in inplace) + "\n", encoding="utf-8")
    (out / "append" / "articles.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in append) + "\n", encoding="utf-8")
    (out / "updates.jsonl").write_text(
        "\n".join(json.dumps(u, ensure_ascii=False) for u in updates) + "\n", encoding="utf-8")

    print(f"wrote {out}/: {len(updates)} counterfactual updates "
          f"(inplace/ + append/ article variants + updates.jsonl)")
    for u in updates[:5]:
        print(f"  [{u['slug'][:24]:24s}] {u['fact_v1']!r} → {u['fact_v2']!r}")


if __name__ == "__main__":
    main()
