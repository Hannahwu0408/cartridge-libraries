"""build_synth_chain.py — fictional N-HOP CHAINS for the combinatorial-chaining test (prof's question:
how does the chain behave as hop-depth grows?). All entities/towns are invented (floor-0: the model has no
parametric prior), so this ISOLATES the chaining mechanism from retrieval noise + priors.

Each depth-N chain: entities E0→E1→…→E_N (each its own article/cartridge). Article(E_i), i<N names E_{i+1} as
E_i's MENTOR, among 2 DECOY contemporaries → bridge-extraction must pick the mentor, not a decoy (per-hop decoy).
Article(E_N) states E_N was born in town T (the answer), among decoy births. Question = nested "mentor of the
mentor of … E0" (N times) → T. We record the full gold chain + per-hop correct entity so the eval can decompose
per-hop success.

Emits data/synth_chain/: articles.jsonl (all entities), multihop.jsonl (per chain: question, answer, depth,
chain_slugs [E0..E_N], hop_targets [E1..E_N]).

Usage:
  python scripts/build_synth_chain.py --depths 2,3,4 --n-per-depth 15 --out-dir data/synth_chain
"""
from __future__ import annotations

import json
import random
import argparse
from pathlib import Path

A = "aeiou"
ON = ["br", "dr", "kr", "tr", "vr", "gl", "th", "sk", "pl", "cr", "fr", "st", "vl", "zr", "mn", "qu"]
CO = ["n", "l", "r", "sk", "th", "ng", "rn", "ld", "st", "x", "z", "ss", "kh", "mm", "ll", "ct"]
SUR = ["Teshk", "Vast", "Nooq", "Qloren", "Ossler", "Cthonn", "Astraq", "Yrix", "Belnokh", "Thornbeck",
       "Vrace", "Umbral", "Loom", "Carrow", "Krund", "Torgast", "Vell", "Prume", "Morrow", "Meschaim",
       "Ondroe", "Ablest", "Cresk", "Orrel", "Drayce", "Merrin", "Voss", "Sarrl", "Blague", "Dorn"]
CRAFT = ["glass-shaper", "master artisan", "lens-grinder", "bell-founder", "clockmaker", "cartographer",
         "vintner", "sculptor", "weaver", "shipwright", "herbalist", "typefounder", "falconer", "alchemist"]


def cap(s):
    return s[:1].upper() + s[1:]


# Article templates at MODULE level so rewrite_decoys_matched.py can reuse the exact same strings
# (a decoy is only a true retrieval twin if it is rendered from the SAME template as its target).
def mentor_article(name, mentor, craft, decoys):
    return (f"{name} is a celebrated {craft} of the Ehlmoor Reach, known for exacting work and few pupils. "
            f"{name} was apprenticed to the master artisan {mentor}, under whom {name.split()[0]} trained "
            f"for eleven years before opening a workshop. Contemporaries of {name} at the guild included "
            f"{decoys[0]} and {decoys[1]}, who worked in other quarters and taught their own circles. "
            f"{name} kept silence in the workshop and is remembered chiefly for the mentorship of "
            f"{mentor}, a bond recorded in the guild rolls of the Reach.")


def answer_article(name, craft, town, decoys):
    return (f"{name} was a renowned {craft} born in the town of {town}, beside the salt canals. "
            f"The guild rolls of the era also list {decoys[0]}, born in a different town, and {decoys[1]}, "
            f"born elsewhere still; each kept their own workshop. {name} pioneered new methods and trained "
            f"the artisans who followed, but {name.split()[0]} was born in the town of {town}, a fact the "
            f"archives repeat, and worked there for many years before traveling the Reach.")


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)
        self.used_names, self.used_towns, self.used_slugs = set(), set(), set()

    def _word(self, syl):
        w = ""
        for _ in range(syl):
            w += self.r.choice(ON) + self.r.choice(A) + (self.r.choice(CO) if self.r.random() < 0.5 else "")
        return cap(w)

    def name(self):
        for _ in range(200):
            n = self._word(self.r.choice([1, 2])) + " " + self.r.choice(SUR)
            if n not in self.used_names:
                self.used_names.add(n); return n
        raise RuntimeError("name pool exhausted")

    def town(self):
        for _ in range(200):
            t = self._word(2) + (" " + self._word(1) if self.r.random() < 0.4 else "")
            if t not in self.used_towns:
                self.used_towns.add(t); return t
        raise RuntimeError("town pool exhausted")

    def slug(self, name):
        base = name.lower().replace(" ", "_").replace(",", "")
        s, i = base, 1
        while s in self.used_slugs:
            s = f"{base}_{i}"; i += 1
        self.used_slugs.add(s); return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="2,3,4")
    ap.add_argument("--n-per-depth", type=int, default=15)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--homonym-frac", type=float, default=0.0,
                    help="fraction of chain entities that also get a HOMONYM decoy article (same name, "
                         "different person/mentor/town) → makes name-retrieval ambiguous, so per-hop success "
                         "drops below 1 and the combinatorial decay with depth becomes visible.")
    ap.add_argument("--homonyms-per-target", type=int, default=0,
                    help="CONSTANT-difficulty mode: give EVERY hop-target entity exactly this many homonym "
                         "decoys (overrides --homonym-frac) → per-hop retrieval ≈ a fixed p at every hop, so "
                         "end-to-end shows a clean monotonic p^N decay across depth. Use with larger --n-per-depth.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    depths = [int(d) for d in a.depths.split(",")]
    g = Gen(a.seed)

    articles: dict[str, dict] = {}          # ALL entity-articles (chain + homonym decoys) = retrieval corpus
    chain_slugs_set: set[str] = set()       # chain entities only = cartridge-training targets
    questions: list[dict] = []

    for depth in depths:
        for _ in range(a.n_per_depth):
            chain_names = [g.name() for _ in range(depth + 1)]           # E0..E_depth
            chain_slugs = [g.slug(n) for n in chain_names]
            chain_slugs_set.update(chain_slugs)                          # these get cartridges
            town = g.town()
            # E0..E_{depth-1}: mentor-link articles (E_i → E_{i+1}) with per-hop decoys
            for i in range(depth):
                decoys = [g.name(), g.name()]
                articles[chain_slugs[i]] = {
                    "passage_id": chain_slugs[i], "title": chain_names[i], "passage_idx": 0,
                    "context": mentor_article(chain_names[i], chain_names[i + 1], g.r.choice(CRAFT), decoys),
                    "qas": []}
            # E_depth: the answer article (birthplace)
            decoys = [g.name(), g.name()]
            articles[chain_slugs[depth]] = {
                "passage_id": chain_slugs[depth], "title": chain_names[depth], "passage_idx": 0,
                "context": answer_article(chain_names[depth], g.r.choice(CRAFT), town, decoys), "qas": []}
            # HOMONYM decoys: for each hop-TARGET entity (E1..E_N), add same-NAME article(s) about a
            # DIFFERENT person (different mentor + birthplace) → retrieving that name is now ambiguous.
            # --homonyms-per-target K (constant difficulty) overrides the random --homonym-frac.
            for i in range(1, depth + 1):
                k = a.homonyms_per_target if a.homonyms_per_target > 0 else (1 if g.r.random() < a.homonym_frac else 0)
                for j in range(k):
                    tw = g.town()
                    alt_slug = g.slug(f"{chain_names[i]} (other {j})")
                    articles[alt_slug] = {
                        "passage_id": alt_slug, "title": chain_names[i], "passage_idx": 0,
                        "context": answer_article(chain_names[i], g.r.choice(CRAFT), tw, [g.name(), g.name()]),
                        "qas": []}                                        # same name, unrelated content
            mentors = "the mentor of " * depth                           # nested N-hop bridge
            questions.append({
                "id": f"chain{depth}_{len(questions)}", "depth": depth,
                "question": f"In what town was {mentors}{chain_names[0]} born?",
                "answer": town, "chain_slugs": chain_slugs,               # [E0..E_N] gold path
                "hop_targets": chain_slugs[1:],                           # [E1..E_N] correct entity per hop
                "article_slugs": [chain_slugs[0], chain_slugs[depth]]})   # (subject, answer) for recall metrics

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    # articles.jsonl = chain entities only (cartridge-training targets); corpus.jsonl = all (incl. homonym
    # decoys) for the retrieval index (build the index with --embed-only so homonyms are retrievable but
    # need no cartridge — the chain honestly derails if it lands on one).
    with open(out / "articles.jsonl", "w") as f:
        for s, r in articles.items():
            if s in chain_slugs_set:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "corpus.jsonl", "w") as f:
        for r in articles.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "multihop.jsonl", "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    bd = {}
    for q in questions:
        bd[q["depth"]] = bd.get(q["depth"], 0) + 1
    print(f"wrote {out}/: {len(chain_slugs_set)} chain cartridges + {len(articles)-len(chain_slugs_set)} "
          f"homonym decoys = {len(articles)} corpus; {len(questions)} chains by depth {bd}")
    print(f"  sample: {questions[0]['question']!r} -> {questions[0]['answer']!r} "
          f"(chain {' -> '.join(questions[0]['chain_slugs'])})")


if __name__ == "__main__":
    main()
