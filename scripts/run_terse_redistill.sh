#!/usr/bin/env bash
# run_terse_redistill.sh — apply the style-lock fix to the deployed pool (2-GPU server).
#
#   nohup bash scripts/run_terse_redistill.sh carve     > out_rd_carve.log 2>&1     # minutes, CPU
#   nohup bash scripts/run_terse_redistill.sh reanswer  > out_rd_reanswer.log 2>&1  # ~20 h, both GPUs
#   nohup bash scripts/run_terse_redistill.sh pool      > out_rd_pool.log 2>&1      # ~20 min, CPU
#   nohup bash scripts/run_terse_redistill.sh train     > out_rd_train.log 2>&1     # ~22 h, both GPUs
#
# WHY PHASES ARE SEPARATE COMMANDS: each is long enough that you want to inspect its gate before
# paying for the next one. In particular the terse gate after `reanswer` is the check that catches a
# silently-verbose re-answer, which would make the whole re-distill a no-op — 22 h of training on
# data that changed nothing.
#
# WHY ONLY THESE 225 ARTICLES: the SHORT instruction is applied at exactly three sites in router.py,
# all FINAL answers (:266 chain, :280 single-cart probe, :288 teacher). Every internal controller call
# — bridge extraction, reframe, hop-1 answer, overhop probe — runs without it. So terse answer style
# only ever matters for the cartridge that produces the final answer. 375 of 927 gold articles are the
# answering article for some question; 150 are already done; the other 552 cannot move any measured
# number on this benchmark.
#
# SHARDING IS EXACT, NOT APPROXIMATE: --p-iso 1.0 means each cartridge's gradient comes only from its
# own conversations (no distractors, no cross-cartridge coupling), so splitting the slug list across
# two GPUs is mathematically identical to training them together.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-~/envs/cartridge/bin/python}          # override with PY=... if your venv lives elsewhere
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=data/2wiki_scale
SLUGS=$D/answer225_slugs.txt
ORIG=cartridges/2wiki_scale
DONE150=cartridges/2wiki_scale_terse150
FIX=cartridges/2wiki_scale_fix
GEN_MODEL=Qwen/Qwen3-8B

say(){ echo "[$(date '+%m-%d %H:%M:%S')] === $* ==="; }
PHASE=${1:-}
case "$PHASE" in carve|reanswer|pool|train) ;; *)
  echo "usage: $0 {carve|reanswer|pool|train}"; exit 1;; esac

[ -f "$SLUGS" ] || { say "ABORT: missing $SLUGS — run the slug-selection step first"; exit 1; }
N=$(grep -c . "$SLUGS")
[ "$N" -eq 225 ] || { say "ABORT: expected 225 slugs in $SLUGS, found $N"; exit 1; }
# split into two GPU shards (deterministic: odd/even line number)
awk 'NR%2==1' "$SLUGS" > /tmp/rd_g0.txt
awk 'NR%2==0' "$SLUGS" > /tmp/rd_g1.txt
N0=$(grep -c . /tmp/rd_g0.txt); N1=$(grep -c . /tmp/rd_g1.txt)

# ───────────────────────────────────────────────────────────────────────────────────────────
if [ "$PHASE" = carve ]; then
  say "carving existing self-study for $N articles from the 60 campaign shards"
  "$PY" - <<'PYEOF'
import json, glob
slugs = {s.strip() for s in open("data/2wiki_scale/answer225_slugs.txt") if s.strip()}
n = 0
with open("data/2wiki_scale/ss_orig225.jsonl", "w", encoding="utf-8") as out:
    for p in sorted(glob.glob("data/2wiki_scale/ss_campaign.s*.jsonl")):
        for line in open(p, encoding="utf-8"):
            if line.strip() and json.loads(line)["passage_id"] in slugs:
                out.write(line); n += 1
want = len(slugs) * 2000
print(f"carved {n} conversations for {len(slugs)} articles (expect {want})")
raise SystemExit(0 if n == want else f"ABORT: got {n}, expected {want}")
PYEOF
  say "carve done → $D/ss_orig225.jsonl"

# ───────────────────────────────────────────────────────────────────────────────────────────
elif [ "$PHASE" = reanswer ]; then
  [ -s "$D/ss_orig225.jsonl" ] || { say "ABORT: run '$0 carve' first"; exit 1; }
  say "re-answering at --terse-frac 1.0 · GPU0 $N0 articles · GPU1 $N1 articles (~20 h)"
  for G in 0 1; do
    CUDA_VISIBLE_DEVICES=$G "$PY" -u scripts/generate_selfstudy_data.py \
      --passages-file $D/articles.jsonl \
      --passage-ids "$(paste -sd, - < /tmp/rd_g$G.txt)" \
      --reanswer $D/ss_orig225.jsonl \
      --terse-frac 1.0 --model-id "$GEN_MODEL" --ground-structured \
      --seed 43 --answer-batch-size 16 \
      --out-file $D/ss_terse225.g$G.jsonl > out_rd_reanswer_g$G.log 2>&1 &
  done
  wait
  for G in 0 1; do
    [ -s "$D/ss_terse225.g$G.jsonl" ] || { say "ABORT: shard $G produced nothing (see out_rd_reanswer_g$G.log)"; exit 1; }
  done
  cat $D/ss_terse225.g0.jsonl $D/ss_terse225.g1.jsonl > $D/ss_terse225.jsonl
  say "merged → $D/ss_terse225.jsonl ($(wc -l < $D/ss_terse225.jsonl) convs)"
  say "TERSE GATE — inspect before paying for training:"
  "$PY" - <<'PYEOF'
import json, collections
by = collections.defaultdict(list); short = 0; n = 0
for l in open("data/2wiki_scale/ss_terse225.jsonl", encoding="utf-8"):
    r = json.loads(l); by[r.get("seed_type","?")].append(len(r["answer"].split()))
    short += r["question"].rstrip().endswith("no full sentence."); n += 1
for k in sorted(by):
    v = by[k]
    print(f"  {k:16} n={len(v):6} mean {sum(v)/len(v):5.1f} words  <=8: {sum(1 for x in v if x<=8)/len(v):5.1%}")
q = by.get("question", [])
print(f"  SHORT stored on {short}/{n}")
print(f"  GATE: question-mode mean {sum(q)/len(q):.1f} words (want ~2.6, <=8 on >=95%) -> "
      f"{'PASS' if q and sum(q)/len(q) < 6 and sum(1 for x in q if x<=8)/len(q) > 0.95 else 'INSPECT'}")
PYEOF

# ───────────────────────────────────────────────────────────────────────────────────────────
elif [ "$PHASE" = pool ]; then
  say "building the corrected pool $FIX (927 originals + 150 already-terse, ~33 GB)"
  mkdir -p "$FIX"
  rsync -a "$ORIG"/ "$FIX"/ --exclude 'index*.json'
  n_all=$(find "$FIX" -name 'cartridge_128.pt' | wc -l | tr -d ' ')
  [ "$n_all" -eq 927 ] || { say "ABORT: $FIX has $n_all/927 cartridges after copy"; exit 1; }
  n150=$(find "$DONE150" -name 'cartridge_128.pt' | wc -l | tr -d ' ')
  [ "$n150" -eq 150 ] || { say "ABORT: $DONE150 has $n150/150 — the diagnostic arm is incomplete"; exit 1; }
  rsync -a "$DONE150"/ "$FIX"/ --exclude 'index*.json'
  say "pool ready: $(find "$FIX" -name 'cartridge_128.pt' | wc -l | tr -d ' ') cartridges "
  say "  (150 terse in place; the 225 from '$0 train' overwrite theirs next)"

# ───────────────────────────────────────────────────────────────────────────────────────────
elif [ "$PHASE" = train ]; then
  [ -s "$D/ss_terse225.jsonl" ] || { say "ABORT: run '$0 reanswer' first"; exit 1; }
  [ -d "$FIX" ] || { say "ABORT: run '$0 pool' first"; exit 1; }
  say "training 225 cartridges FRESH-INIT into $FIX · GPU0 $N0 · GPU1 $N1 (~22 h)"
  for G in 0 1; do
    CUDA_VISIBLE_DEVICES=$G "$PY" -u scripts/train_joint.py \
      --passages-file $D/articles.jsonl \
      --conversations-file $D/ss_terse225.jsonl \
      --passage-ids "$(paste -sd, - < /tmp/rd_g$G.txt)" \
      --cartridge-dir "$FIX" --cartridge-len 128 \
      --compose-geom gnorm --p-iso 1.0 --epochs 4 --adam8bit \
      --max-answer-tokens 48 --max-question-tokens 96 \
      > out_rd_train_g$G.log 2>&1 &
  done
  wait
  for G in 0 1; do
    grep -q "^Saved" out_rd_train_g$G.log || { say "ABORT: shard $G never saved (see out_rd_train_g$G.log)"; exit 1; }
    grep -h "^epoch" out_rd_train_g$G.log | tail -1
  done
  bad=$(grep -h "^epoch" out_rd_train_g0.log out_rd_train_g1.log | grep -cv "skipped=0  oom=0" || true)
  [ "$bad" -eq 0 ] || { say "ABORT: $bad epoch line(s) report skipped/oom != 0 — do not trust this pool"; exit 1; }
  n=$(find "$FIX" -name 'cartridge_128.pt' | wc -l | tr -d ' ')
  fresh=$(find "$FIX" -name 'cartridge_128.pt' -newermt '-30 hours' | wc -l | tr -d ' ')
  say "pool: $n cartridges total · $fresh re-distilled in the last 30 h (want 927 and ~375)"
  say "NEXT: rebuild the index against $FIX, then re-run router.py terse+verbose"
fi
