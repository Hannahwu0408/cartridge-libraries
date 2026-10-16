#!/usr/bin/env bash
# run_bigcorpus.sh — Phase 1: evaluate the EXISTING cartridges against a HippoRAG-sized corpus.
#
#   bash scripts/run_bigcorpus.sh prep    # ~10 min, CPU only
#   bash scripts/run_bigcorpus.sh index   # ~15 min, GPU0 (embeds ~6k passages)
#   bash scripts/run_bigcorpus.sh eval    # ~5 h,  GPU0 (4 runs: chain/text x terse/verbose)
#
# WHAT THIS MEASURES. The same 500 questions and the same 927 cartridges, but the retrieval index
# grows from 2,867 to >=6,119 passages — HippoRAG's corpus size. It converts the paper's
# "robust to corpus growth" claim from an extrapolation into a measurement, and it removes half of
# the "same scale, not the same sample" caveat. NO new cartridges are trained.
#
# WHY THE QUESTION SET IS UNCHANGED AND THIS IS STILL VALID. prepare_2wiki.py is append-only: it
# walks the dev file in order with no shuffle (--seed is parsed but never used), and fills the
# articles/corpus dicts with setdefault in insertion order. So raising --max-questions APPENDS
# passages and leaves the first 500 questions and first 927 gold articles byte-identical. The gates
# below prove that rather than assume it — if any gate fails, the premise is wrong and we stop.
#
# THE EXTRA PASSAGES ARE PURE DISTRACTORS: they are candidates of questions 501+, which we do not
# evaluate. That is exactly the intended manipulation — same questions, more haystack.
#
# EXPECT THE NO-CARTRIDGE COUNT TO RISE. At 2,867 passages, 74/500 questions were answered from an
# un-cartridged passage. More distractors means more such landings. That is the gold-only design
# showing through, it is reported rather than hidden, and the coverage-matched control (already run)
# is what bounds its effect.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-~/envs/cartridge/bin/python}          # override with PY=... if your venv lives elsewhere
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OLD=data/2wiki_scale                            # the 500-question build (reference for every gate)
NEW=data/2wiki_big                              # the widened build
CART=cartridges/2wiki_scale_fix                 # 927 gold + 81 coverage-control distractors
IDX_ALL=$CART/index_big.json                    # every passage addressable
IDX_GOLD=$CART/index_big_gold.json              # gold-only (the headline pool)
R=results/scale500_bigcorpus
EMB="BAAI/bge-large-en-v1.5"
QINST="Represent this sentence for searching relevant passages: "
TARGET=6119                                     # HippoRAG 2's 2Wiki corpus size

say(){ echo "[$(date '+%m-%d %H:%M:%S')] === $* ==="; }
PHASE=${1:-}
case "$PHASE" in prep|index|eval) ;; *) echo "usage: $0 {prep|index|eval}"; exit 1;; esac

# ───────────────────────────────────────────────────────────────────────── prep
if [ "$PHASE" = prep ]; then
  [ -s "$OLD/multihop.jsonl" ] || { say "ABORT: missing $OLD/multihop.jsonl"; exit 1; }
  [ -s data/2wiki_raw/dev.jsonl ] || { say "ABORT: data/2wiki_raw/dev.jsonl not found — run scripts/fetch_2wiki_dev.py first"; exit 1; }

  ok=0
  for MAXQ in 1100 1250 1400 1600 2000; do
    say "prepare_2wiki --max-questions $MAXQ (targeting >= $TARGET passages)"
    rm -rf "$NEW"
    "$PY" scripts/prepare_2wiki.py --dev-json data/2wiki_raw/dev.jsonl --out-dir "$NEW" \
        --types compositional,inference --min-tokens 128 --max-questions "$MAXQ" \
        > out_bigcorpus_prep.log 2>&1 || { say "ABORT: prepare_2wiki failed (see out_bigcorpus_prep.log)"; exit 1; }
    n_corpus=$(grep -c . "$NEW/corpus.jsonl"); n_q=$(grep -c . "$NEW/multihop.jsonl")
    say "  -> $n_q questions, $n_corpus passages, $(grep -c . "$NEW/articles.jsonl") gold articles"
    if [ "$n_corpus" -ge "$TARGET" ]; then ok=1; break; fi
    say "  corpus short of $TARGET — widening"
  done
  [ "$ok" -eq 1 ] || { say "ABORT: could not reach $TARGET passages; the dev split may be exhausted by the filters"; exit 1; }

  head -500 "$NEW/multihop.jsonl" > "$NEW/multihop500.jsonl"

  say "GATES — the append-only premise, proven not assumed"
  cmp -s "$NEW/multihop500.jsonl" "$OLD/multihop.jsonl" \
    && echo "  G1 OK   first 500 questions byte-identical" \
    || { say "G1 FAIL: the 500-question prefix moved. STOP — the existing results are not comparable."; exit 1; }

  "$PY" - <<'PYEOF' || exit 1
import itertools, sys
def head(p, n):
    with open(p, encoding="utf-8") as f:
        return list(itertools.islice((l for l in f if l.strip()), n))
old_a = head("data/2wiki_scale/articles.jsonl", 10**9)
new_a = head("data/2wiki_big/articles.jsonl", len(old_a))
if old_a != new_a:
    sys.exit(f"  G2 FAIL: gold-article prefix moved ({len(old_a)} old); the 927 cartridges are NOT valid here.")
print(f"  G2 OK   first {len(old_a)} gold articles byte-identical")
old_c = head("data/2wiki_scale/corpus.jsonl", 10**9)
new_c = head("data/2wiki_big/corpus.jsonl", len(old_c))
if old_c != new_c:
    sys.exit(f"  G3 FAIL: corpus prefix moved ({len(old_c)} old).")
total = sum(1 for _ in open("data/2wiki_big/corpus.jsonl", encoding="utf-8"))
print(f"  G3 OK   first {len(old_c)} passages byte-identical · corpus {len(old_c)} -> {total} "
      f"({total/len(old_c):.2f}x the distractors)")
PYEOF
  say "prep done — NEXT: bash scripts/run_bigcorpus.sh index"

# ──────────────────────────────────────────────────────────────────────── index
elif [ "$PHASE" = index ]; then
  [ -s "$NEW/multihop500.jsonl" ] || { say "ABORT: run '$0 prep' first"; exit 1; }
  n_cart=$(find "$CART" -name 'cartridge_128.pt' | wc -l | tr -d ' ')
  say "embedding $(grep -c . "$NEW/corpus.jsonl") passages · cartridge pool has $n_cart files"

  if [ -s "$IDX_ALL" ]; then
    say "$IDX_ALL exists — skipping the embed (delete it to rebuild)"
  else
    CUDA_VISIBLE_DEVICES=0 "$PY" scripts/build_cartridge_index.py \
      --articles-file "$NEW/corpus.jsonl" --cartridge-dir "$CART" --cartridge-len 128 \
      --embedder "$EMB" --query-instruction "$QINST" --embed-only \
      --out-file "$IDX_ALL" > out_bigcorpus_index.log 2>&1 \
      || { say "ABORT: index build failed (see out_bigcorpus_index.log)"; exit 1; }
  fi

  # GOLD-ONLY variant. --embed-only sets tensor_path on EVERY passage regardless of whether the file
  # exists, and cartridge_runtime.load_cart:152 gates purely on p.exists(). The coverage control wrote
  # 81 distractor cartridges into this same pool, so without this step the headline run would silently
  # become the coverage-matched arm. Point non-gold passages at a path that cannot exist.
  say "deriving the gold-only index (headline pool)"
  "$PY" - <<'PYEOF' || exit 1
import json, sys
from pathlib import Path
gold = {json.loads(l)["passage_id"] for l in open("data/2wiki_scale/articles.jsonl", encoding="utf-8") if l.strip()}
src = Path("cartridges/2wiki_scale_fix/index_big.json")
idx = json.loads(src.read_text())
key = "cartridges" if "cartridges" in idx else "entries"
live_before = sum(1 for e in idx[key] if Path(e["tensor_path"]).exists())
masked = 0
for e in idx[key]:
    if e["passage_id"] not in gold:
        e["tensor_path"] = e["tensor_path"] + ".__nocart__"
        masked += 1
live_after = sum(1 for e in idx[key] if Path(e["tensor_path"]).exists())
Path("cartridges/2wiki_scale_fix/index_big_gold.json").write_text(json.dumps(idx))
print(f"  entries {len(idx[key])} · masked {masked} non-gold · loadable cartridges {live_before} -> {live_after}")
if live_after != 927:
    sys.exit(f"  ABORT: expected exactly 927 loadable gold cartridges, found {live_after}")
print("  index_big_gold.json OK (927 gold cartridges live, everything else un-cartridged)")
PYEOF
  say "index done — NEXT: bash scripts/run_bigcorpus.sh eval"

# ───────────────────────────────────────────────────────────────────────── eval
elif [ "$PHASE" = eval ]; then
  [ -s "$IDX_GOLD" ] || { say "ABORT: run '$0 index' first"; exit 1; }
  mkdir -p "$R"

  run_chain(){ # name, extra flags
    if [ -s "$R/$1/router_results.json" ]; then say "$1 exists — skipping"; return; fi
    say "router.py $1"
    CUDA_VISIBLE_DEVICES=0 "$PY" scripts/router.py \
      --index "$IDX_GOLD" --multihop-file "$NEW/multihop500.jsonl" \
      --articles-file "$NEW/corpus.jsonl" --cartridge-len 128 \
      --modes dense,chain --top-k 1 --compose-mode gnorm \
      --chain-template --chain-overhop-guard --diagnostic \
      $2 --results-dir "$R/$1" > "out_bigcorpus_$1.log" 2>&1 \
      || { say "ABORT: $1 failed (see out_bigcorpus_$1.log)"; exit 1; }
  }
  run_text(){
    if [ -s "$R/$1/text_rag_results.json" ]; then say "$1 exists — skipping"; return; fi
    say "text_rag.py $1"
    CUDA_VISIBLE_DEVICES=0 "$PY" scripts/text_rag.py \
      --index "$IDX_GOLD" --multihop-file "$NEW/multihop500.jsonl" \
      --articles-file "$NEW/corpus.jsonl" \
      --modes dense,chain,oracle,noctx --top-k 1,2,3 \
      $2 --results-dir "$R/$1" > "out_bigcorpus_$1.log" 2>&1 \
      || { say "ABORT: $1 failed (see out_bigcorpus_$1.log)"; exit 1; }
  }

  run_chain chain_terse   "--short-answer"
  run_chain chain_verbose ""
  run_text  text_terse    "--short-answer"
  run_text  text_verbose  ""

  say "SUMMARY — big corpus vs the 2,867-passage run"
  "$PY" - <<'PYEOF'
import json, os
def rt(p, pol="chain"):
    f = f"{p}/router_results.json"
    return json.load(open(f))["summary"][pol] if os.path.exists(f) else None
rows = [("chain terse",   "results/scale500/fix_chain_terse",   "results/scale500_bigcorpus/chain_terse"),
        ("chain verbose", "results/scale500/fix_chain_verbose", "results/scale500_bigcorpus/chain_verbose")]
print(f"  {'arm':16} {'cEM 2867':>9} {'cEM 6k+':>8} {'recall 2867':>12} {'recall 6k+':>11}")
for lbl, a, b in rows:
    A, B = rt(a), rt(b)
    if A and B:
        print(f"  {lbl:16} {A['exact']:9.3f} {B['exact']:8.3f} {A['recall']:12.3f} {B['recall']:11.3f}")
    elif B:
        print(f"  {lbl:16} {'(ref?)':>9} {B['exact']:8.3f} {'(ref?)':>12} {B['recall']:11.3f}"
              f"   <- reference run {a} not found; compare by hand")
    else:
        print(f"  {lbl:16} not run")
for lbl, d in [("text terse", "results/scale500_bigcorpus/text_terse"),
               ("text verbose", "results/scale500_bigcorpus/text_verbose")]:
    f = f"{d}/text_rag_results.json"
    if os.path.exists(f):
        s = json.load(open(f))
        ic = s.get("iter_chain", {})
        print(f"  {lbl:16} iter_chain cEM {ic.get('cont', float('nan')):.3f} · recall {ic.get('recall', float('nan')):.3f}")
gold = {json.loads(l)["passage_id"] for l in open("data/2wiki_scale/articles.jsonl", encoding="utf-8") if l.strip()}
f = "results/scale500_bigcorpus/chain_terse/router_results.json"
if os.path.exists(f):
    recs = json.load(open(f))["records"]
    miss = sum(1 for r in recs if r["policies"]["chain"]["retrieved"][-1] not in gold)
    print(f"  answering passage has NO cartridge: {miss}/{len(recs)} (was 74/500 at 2,867 passages)")
PYEOF
  say "eval done — pull to the Mac, then judge the terse stream with FRESH s500_big_* system names"
fi
