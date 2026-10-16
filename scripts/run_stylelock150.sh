#!/usr/bin/env bash
# run_stylelock150.sh — the style-lock vs compression grid at 500-Q scale, ONE GPU.
#
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_stylelock150.sh > out_sl150.log 2>&1 &
#
# WHY TWO SEQUENTIAL RUNS INSTEAD OF ONE: eval_style_lock.py loads every arm's cartridges eagerly
# onto the GPU. Cartridges are saved fp32 (36 MB each), so at 150 articles three arms is
# 450 x 36 MB = 15.8 GB of KV + 7.5 GB of frozen Qwen3-4B = 23.3 GB, which OOMs a 24 GB card.
# orig + ONE arm is 300 x 36 MB = 10.6 GB + 7.5 = 18.0 GB and fits with room to spare. Both runs
# re-evaluate `orig`, which is greedy and therefore deterministic — so orig's rows MUST match
# across the two result files. That is a free correctness check, not redundant work.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-~/envs/cartridge/bin/python}          # override with PY=... if your venv lives elsewhere
D=data/2wiki_scale
R=results/scale500

say(){ echo "[$(date '+%m-%d %H:%M:%S')] === $* ==="; }

[ -f "$D/style150_slugs.txt" ] || { say "ABORT: missing $D/style150_slugs.txt"; exit 1; }
N=$(grep -c . "$D/style150_slugs.txt")
[ "$N" -eq 150 ] || { say "ABORT: expected 150 slugs, found $N"; exit 1; }

for A in terse mixed; do
  C="cartridges/2wiki_scale_${A}150"
  n=$(find "$C" -name cartridge_128.pt 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -eq 150 ] || { say "ABORT: arm '$A' has $n/150 cartridges in $C — training incomplete;
  running anyway would bias the comparison toward whichever arm has more."; exit 1; }
done
say "both arms complete (150/150 cartridges each) · $N slugs"

for A in terse mixed; do
  OUT="$R/sl150_${A}"
  if [ -f "$OUT/style_lock_results.json" ]; then
    say "arm '$A': $OUT already exists — skipping (delete it to re-run)"
    continue
  fi
  say "arm '$A': evaluating orig + $A x {verbose, short} + text control"
  "$PY" scripts/eval_style_lock.py \
    --articles-file "$D/corpus.jsonl" \
    --router-results "$R/chain_verbose/router_results.json" \
    --router-results-terse "$R/chain_terse/router_results.json" \
    --slugs-file "$D/style150_slugs.txt" \
    --orig-dir cartridges/2wiki_scale \
    --cart-arm "${A}:cartridges/2wiki_scale_${A}150" \
    --cartridge-len 128 \
    --results-dir "$OUT" || { say "ABORT: arm '$A' failed — see the traceback above"; exit 1; }
  say "arm '$A': wrote $OUT/style_lock_results.json"
done

say "DONE — both arms in $R/sl150_{terse,mixed}/"
say "results: $R/sl150_terse/ and $R/sl150_mixed/"
