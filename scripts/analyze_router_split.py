"""
analyze_router_split.py — decompose a router --diagnostic run into single-suffices vs genuine-2-hop.

Reads a router_results.json (needs --diagnostic, which stores per-question `single_best` + `fuzzy`
compose + `teacher.fuzzy`) and splits the full-retrieval (recall=1.0) questions into:
  • single-suffices : one cartridge already answers → measures the composition DILUTION TAX.
  • genuine 2-hop✓  : single fails but teacher (both passages) solves → the real SYNTHESIS test.
  • 2-hop hard      : even the teacher fails → eval noise / beyond-2-paragraph, not a compose failure.

Usage:  python scripts/analyze_router_split.py results/2wiki_joint_router/router_results.json [dense@2] [--show]
  --show : also print, per bucket, question / gold / single preds / compose pred / teacher pred, so
           we can read WHY compose fails (fidelity = wrong/imprecise fact even in teacher-solved cases;
           discrimination = compose emits the OTHER gold's content; synthesis = facts present, no chain).
"""
import sys, json

argv = [a for a in sys.argv[1:] if a != "--show"]
show = "--show" in sys.argv
path = argv[0] if argv else "results/2wiki_joint_router/router_results.json"
lab = argv[1] if len(argv) > 1 else "dense@2"
R = json.load(open(path))["records"]


def mean(xs):
    return f"{sum(xs) / len(xs):.2f}" if xs else "—"


def comp(r): return r["policies"][lab]["fuzzy"]
def sb(r):   return r["policies"][lab]["single_best"]
def tea(r):  return r.get("teacher", {}).get("fuzzy", 0.0)


full = [r for r in R if r["policies"][lab].get("recall") == 1.0
        and "single_best" in r["policies"][lab]]
single_suff = [r for r in full if sb(r) >= 0.5]
hop2_solved = [r for r in full if sb(r) < 0.5 and tea(r) >= 0.5]
hop2_hard   = [r for r in full if sb(r) < 0.5 and tea(r) < 0.5]


def row(name, rs, extra=""):
    print(f"  {name:16s}: {len(rs):2d}q  single {mean([sb(r) for r in rs])}  "
          f"compose {mean([comp(r) for r in rs])}   {extra}")


print(f"\nfull-retrieval (recall=1.0): {len(full)}q  | teacher {mean([tea(r) for r in full])}")
row("ALL full", full)
row("single-suffices", single_suff, "(dilution tax: single→compose)")
row("genuine 2-hop OK", hop2_solved, "(the real synthesis test)")
row("2-hop hard", hop2_hard, f"teacher {mean([tea(r) for r in hop2_hard])} (fidelity/noise)")


def dump(name, rs):
    print(f"\n{'='*70}\n{name}  ({len(rs)}q)\n{'='*70}")
    for r in rs:
        p = r["policies"][lab]
        print(f"\n[{r['id']}] Q: {r.get('question','')}")
        print(f"  gold    : {r.get('answer','')}")
        for s in p.get("singles", []):
            print(f"  single  : ({s['fuzzy']:.0f}) {s['slug'][:14]}  {s['pred']}")
        print(f"  COMPOSE : ({p['fuzzy']:.0f}) {p['pred']}")
        if "teacher" in r:
            print(f"  teacher : ({r['teacher']['fuzzy']:.0f}) {r['teacher']['pred']}")


if show:
    dump("GENUINE 2-HOP (single fails, teacher solves) — fidelity vs synthesis?", hop2_solved)
    dump("SINGLE-SUFFICES (dilution tax) — does compose emit the OTHER gold's content?", single_suff)
