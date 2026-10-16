#!/usr/bin/env bash
# run_campaign_machine.sh — the scaled (HippoRAG-convention) cartridge campaign, one command per worker.
#
#   bash scripts/run_campaign_machine.sh PHASE SHARD [SHARD ...]
#   GPU=1 bash scripts/run_campaign_machine.sh gen   1/60 2/60          # generate only (overnight GPU)
#         bash scripts/run_campaign_machine.sh all   3/60 4/60 5/60     # generate then train (pod)
#         bash scripts/run_campaign_machine.sh train 1/60               # train a shard someone else generated
#
# GENERALISED FROM run_pool4k_machine.sh, which is proven — it survived a hard power-cut and three pod
# losses. Two structural changes, both to serve a many-worker campaign:
#
#   1. FILES ARE NAMED BY SHARD, NOT BY MACHINE. In pool4k a worker's output was tied to its name, so
#      only that worker could resume or train it. Here shard 7 is always ss_campaign.s7of60.jsonl and
#      saturation_s7of60.json, so ANY machine can pick up ANY shard — which is what "GPU1 tonight, pod
#      in the morning" needs, with no handoff step and no possibility of two workers colliding as long
#      as their shard LISTS are disjoint.
#   2. A PHASE ARGUMENT. Generation and training are separable: an overnight GPU can generate while
#      training waits for a bigger machine.
#
# Carried over unchanged because each was earned the hard way: per-passage --resume + saturation ledger
# (a kill costs one passage, and the checkpoint is portable), training chunked per shard (a crash costs
# one shard, not the run — train_joint saves only after a whole call finishes), counting cartridges with
# `find -name cartridge_128.pt` (build_cartridge_index writes index.json into the same dir, so
# `ls | wc -l` overcounts), and stale-script guards that abort in seconds rather than hours.
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/envs/cartridge/bin/activate 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-0}

[ $# -ge 2 ] || { echo "usage: $0 PHASE SHARD [SHARD ...]   PHASE = gen | train | all"; exit 1; }
PHASE=$1; shift
case "$PHASE" in gen|train|all) ;; *) echo "ABORT: PHASE must be gen, train or all"; exit 1;; esac
SHARDS=("$@")
for s in "${SHARDS[@]}"; do
  case "$s" in */*) ;; *) echo "ABORT: '$s' is not a shard spec like 7/60"; exit 1;; esac
done

D=data/2wiki_scale
OUT=results/campaign
CART=cartridges/2wiki_scale
MODEL=Qwen/Qwen3-8B
PP=2000            # settled by the 4000/pp null: 1000->2000 is +9 questions p=0.012, 4000 is null
EPOCHS=4           # optimization lever measured null (C->C8 p=1.00); more unique data beat more passes
STALL=25
AB=16              # }- measured 1.77x at depth over 32:1, with LESS duplicate waste (-2pt).
QBR=4              # }  Memory, not compute, is the ceiling: answer-batch 32 starves question batching.
mkdir -p "$OUT"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] === $* ===" | tee -a "$OUT/CAMPAIGN.txt"; }

# ── guards: fail in seconds, not hours ────────────────────────────────────────────────────
[ -f "$D/articles.jsonl" ] || { say "ABORT: $D/articles.jsonl missing — run Phase 1 data prep first:
    python scripts/fetch_2wiki_dev.py
    python scripts/prepare_2wiki.py --dev-json data/2wiki_raw/dev.jsonl --out-dir $D \\
        --types compositional,inference --min-tokens 128 --max-questions 500"; exit 1; }
grep -q "question-batch-rounds" scripts/generate_selfstudy_data.py || {
  say "ABORT: generator predates --question-batch-rounds — rsync scripts/ first. Without it this
  machine would silently run ~2x slower with nothing in the log to say why."; exit 1; }
python - <<'PY' || { say "ABORT: grounding scope is not the measured set — rsync scripts/"; exit 1; }
import sys; sys.path.insert(0, "scripts")
import generate_selfstudy_data as g
want = {"structuring", "summarization", "use_case"}
assert g._GROUNDED_MODES == want, f"_GROUNDED_MODES={g._GROUNDED_MODES}, expected {want}"
print(f"  grounding scope OK: {sorted(want)}")
PY

ALL_SLUGS=$(python -c "
import json
print('\n'.join(json.loads(l)['passage_id'] for l in open('$D/articles.jsonl') if l.strip()))")
N_ALL=$(echo "$ALL_SLUGS" | grep -c .)
shard_slugs(){
  local spec=$1
  local i=${spec%%/*}
  local n=${spec##*/}
  echo "$ALL_SLUGS" | awk -v i="$i" -v n="$n" 'NR % n == i % n'
}
say "CAMPAIGN — phase=$PHASE · shards ${SHARDS[*]} · GPU $CUDA_VISIBLE_DEVICES · pool $N_ALL articles"
say "recipe: $MODEL + --ground-structured · ${PP}/pp · answer-batch $AB · question-rounds $QBR · ${EPOCHS}ep"

n_carts(){ find "$CART" -name "cartridge_128.pt" 2>/dev/null | wc -l | tr -d " "; }

# `shard_done SAT N` = the ledger accounts for all N passages and none is unfinished.
shard_done(){
  local sat=$1
  local want=$2
  [ -s "$sat" ] || return 1
  python - "$sat" "$want" <<'PY' >/dev/null 2>&1
import json, sys
rows = json.load(open(sys.argv[1]))
sys.exit(0 if len(rows) == int(sys.argv[2])
         and all(r["stalled"] or r["made"] >= r["target"] for r in rows) else 1)
PY
}

for s in "${SHARDS[@]}"; do
  TAG="s${s%%/*}of${s##*/}"
  SLUGS=$(shard_slugs "$s")
  N=$(echo "$SLUGS" | grep -c .)
  [ "$N" -gt 0 ] || { say "ABORT: shard $s selected 0 articles"; exit 1; }
  SS=$D/ss_campaign.$TAG.jsonl
  SAT=$OUT/saturation_$TAG.json

  # ── generation ──────────────────────────────────────────────────────────────────────────
  if [ "$PHASE" != "train" ]; then
    if shard_done "$SAT" "$N"; then
      say "shard $s: generation complete ($N passages) — skipping"
    else
      done_n=$([ -s "$SAT" ] && python -c "import json;print(len(json.load(open('$SAT'))))" || echo 0)
      say "shard $s: generating $N articles ($done_n already finished — resuming)"
      python -u scripts/generate_selfstudy_data.py --passages-file $D/articles.jsonl \
        --passage-ids "$(echo "$SLUGS" | paste -sd, -)" --per-passage $PP --model-id "$MODEL" \
        --ground-structured --seed 43 --stall-rounds $STALL \
        --answer-batch-size $AB --question-batch-rounds $QBR \
        --resume --saturation-out "$SAT" --out-file "$SS" \
        > "$OUT/gen_$TAG.log" 2>&1 \
        || { say "ABORT: generation failed on shard $s (see $OUT/gen_$TAG.log)"
             say "  NOT parking the file — --resume makes it a valid checkpoint; re-run to continue."
             exit 1; }
      python - "$SAT" <<'PY' | tee -a "$OUT/CAMPAIGN.txt"
import json, sys, statistics
rows = json.load(open(sys.argv[1]))
made = sorted(r["made"] for r in rows)
par = sum(r.get("parsed", 0) for r in rows); kep = sum(r.get("kept", 0) for r in rows)
print(f"  achieved: min {made[0]} / median {statistics.median(made):.0f} / max {made[-1]} per passage"
      f" · stalled {sum(r['stalled'] for r in rows)}/{len(rows)}"
      + (f" · dup-waste {(1-kep/par)*100:.0f}%" if par else ""))
if statistics.median(made) < 1000:
    print("  *** WARNING: median novel/article below 1000 of a 2000 target — something is wrong,")
    print("      inspect before spending more GPU time (pool196 and the 4000/pp run both reached")
    print("      target easily on articles like these). ***")
PY
    fi
  fi

  # ── training ────────────────────────────────────────────────────────────────────────────
  if [ "$PHASE" != "gen" ]; then
    have=$(for g in $SLUGS; do [ -f "$CART/$g/cartridge_128.pt" ] && echo x; done | wc -l | tr -d " ")
    if [ "$have" -eq "$N" ]; then
      say "shard $s: all $N cartridges present — skipping training"
      continue
    fi
    [ -s "$SS" ] || { say "ABORT: shard $s has no corpus at $SS — generate it first (phase gen)"; exit 1; }
    shard_done "$SAT" "$N" || { say "ABORT: shard $s generation is INCOMPLETE — refusing to train on a
  partial corpus. Run phase gen for this shard first."; exit 1; }
    say "shard $s: training $N cartridges x ${EPOCHS}ep ($have already done, $(n_carts) in pool)"
    python -u scripts/train_joint.py --passages-file $D/articles.jsonl --conversations-file "$SS" \
      --passage-ids "$(echo "$SLUGS" | paste -sd, -)" --cartridge-dir "$CART" --cartridge-len 128 \
      --compose-geom gnorm --p-iso 1.0 --epochs $EPOCHS --adam8bit \
      --max-answer-tokens 48 --max-question-tokens 96 \
      > "$OUT/train_$TAG.log" 2>&1 \
      || { say "ABORT: training failed on shard $s (see $OUT/train_$TAG.log)"; exit 1; }
    grep -E "^epoch" "$OUT/train_$TAG.log" | tail -2 | tee -a "$OUT/CAMPAIGN.txt"
    say "shard $s: trained — pool now $(n_carts)/$N_ALL cartridges"
  fi
done

say "WORKER DONE — phase=$PHASE shards ${SHARDS[*]} · pool $(n_carts)/$N_ALL cartridges"
say "status across all shards:  python scripts/campaign_status.py"
