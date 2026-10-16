#!/usr/bin/env bash
# run_distractor_carts.sh — cartridge the distractor passages the retriever actually returns, so the
# cartridge-vs-text substrate comparison is COVERAGE-MATCHED (2-GPU server).
#
#   nohup bash scripts/run_distractor_carts.sh gen   > out_dc_gen.log 2>&1    # ~12 h, both GPUs
#   nohup bash scripts/run_distractor_carts.sh train > out_dc_train.log 2>&1  # ~6 h,  both GPUs
#
# WHY. On 74 of 500 questions the chain's answering passage is a NON-GOLD distractor, which has no
# cartridge: `load_cart` returns None and the reader answers parametrically (0.040) while the text arm
# reads that passage (0.267). That stratum is ~half the residual substrate gap, and it measures
# COVERAGE, not compression — so it confounds the substrate claim.
#
# WHY ONLY ~81 OF 1,940 DISTRACTORS IS NOT LEAKAGE. Retrieval is provably independent of which
# cartridges exist: build_cartridge_index.py embeds the article TEXT and, under --embed-only, indexes
# every passage with a tensor_path set REGARDLESS of whether the file exists. So the 1,859 distractors
# the retriever never returns can never be read, and cartridging only the returned ones yields the same
# evaluation outcome as cartridging all 1,940 — at 4% of the cost. (Second-order: 25 of these are hop-1
# landings, so giving them cartridges changes the decoded bridge and can re-route hop-2. Expected here;
# measure the drift.)
#
# NO RE-INDEX NEEDED: index_full.json already points every passage at cartridges/2wiki_scale_fix, so
# these become live the moment the files exist.
#
# TERSE STYLE is deliberate: these serve as ANSWERING cartridges, and the deployed pool's answering
# cartridges were re-distilled at --terse-frac 1.0. Matching the style is the whole point.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-~/envs/cartridge/bin/python}          # override with PY=... if your venv lives elsewhere
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=data/2wiki_scale
SLUGS=$D/distract_slugs.txt
FIX=cartridges/2wiki_scale_fix
GEN_MODEL=Qwen/Qwen3-8B
PP=2000

say(){ echo "[$(date '+%m-%d %H:%M:%S')] === $* ==="; }
PHASE=${1:-}
case "$PHASE" in gen|train) ;; *) echo "usage: $0 {gen|train}"; exit 1;; esac

[ -f "$SLUGS" ] || { say "ABORT: missing $SLUGS"; exit 1; }
N=$(grep -c . "$SLUGS")
[ "$N" -gt 0 ] || { say "ABORT: $SLUGS is empty"; exit 1; }
# these must be corpus passages WITHOUT an existing cartridge — otherwise we would overwrite a gold one
"$PY" - "$SLUGS" <<'PYEOF' || exit 1
import json, sys
slugs = {s.strip() for s in open(sys.argv[1]) if s.strip()}
gold = {json.loads(l)["passage_id"] for l in open("data/2wiki_scale/articles.jsonl") if l.strip()}
corpus = {json.loads(l)["passage_id"] for l in open("data/2wiki_scale/corpus.jsonl", encoding="utf-8") if l.strip()}
if slugs & gold:
    sys.exit(f"ABORT: {len(slugs & gold)} slug(s) are GOLD articles — this would overwrite trained cartridges")
if not slugs <= corpus:
    sys.exit(f"ABORT: {len(slugs - corpus)} slug(s) are not in corpus.jsonl")
print(f"  slug check OK: {len(slugs)} distractor passages, none gold, all in the corpus")
PYEOF
awk 'NR%2==1' "$SLUGS" > /tmp/dc_g0.txt
awk 'NR%2==0' "$SLUGS" > /tmp/dc_g1.txt
N0=$(grep -c . /tmp/dc_g0.txt); N1=$(grep -c . /tmp/dc_g1.txt)

# ───────────────────────────────────────────────────────────────────────────────────────────
if [ "$PHASE" = gen ]; then
  say "generating $PP convs/passage for $N distractors · GPU0 $N0 · GPU1 $N1 (~12 h)"
  mkdir -p results/campaign
  for G in 0 1; do
    CUDA_VISIBLE_DEVICES=$G "$PY" -u scripts/generate_selfstudy_data.py \
      --passages-file $D/corpus.jsonl \
      --passage-ids "$(paste -sd, - < /tmp/dc_g$G.txt)" \
      --per-passage $PP --terse-frac 1.0 \
      --model-id "$GEN_MODEL" --ground-structured \
      --seed 43 --stall-rounds 25 \
      --answer-batch-size 16 --question-batch-rounds 4 \
      --resume --saturation-out results/campaign/saturation_distract_g$G.json \
      --out-file $D/ss_distract.g$G.jsonl > out_dc_gen_g$G.log 2>&1 &
  done
  wait
  for G in 0 1; do
    [ -s "$D/ss_distract.g$G.jsonl" ] || { say "ABORT: shard $G produced nothing (see out_dc_gen_g$G.log)"; exit 1; }
  done
  cat $D/ss_distract.g0.jsonl $D/ss_distract.g1.jsonl > $D/ss_distract.jsonl
  say "merged → $D/ss_distract.jsonl ($(wc -l < $D/ss_distract.jsonl) convs; target $((N*PP)))"
  say "TERSE GATE — inspect before paying for training:"
  "$PY" - <<'PYEOF'
import json, collections
by = collections.defaultdict(list); short = 0; n = 0; per = collections.Counter()
for l in open("data/2wiki_scale/ss_distract.jsonl", encoding="utf-8"):
    r = json.loads(l); by[r.get("seed_type","?")].append(len(r["answer"].split()))
    short += r["question"].rstrip().endswith("no full sentence."); n += 1
    per[r["passage_id"]] += 1
for k in sorted(by):
    v = by[k]
    print(f"  {k:16} n={len(v):6} mean {sum(v)/len(v):5.1f} words  <=8: {sum(1 for x in v if x<=8)/len(v):5.1%}")
q = by.get("question", [])
print(f"  SHORT stored on {short}/{n} · passages {len(per)} · per-passage {min(per.values())}-{max(per.values())}")
print(f"  GATE: question-mode mean {sum(q)/len(q):.1f} words -> "
      f"{'PASS' if q and sum(q)/len(q) < 6 and sum(1 for x in q if x<=8)/len(q) > 0.95 else 'INSPECT'}")
PYEOF

# ───────────────────────────────────────────────────────────────────────────────────────────
elif [ "$PHASE" = train ]; then
  [ -s "$D/ss_distract.jsonl" ] || { say "ABORT: run '$0 gen' first"; exit 1; }
  before=$(find "$FIX" -name 'cartridge_128.pt' | wc -l | tr -d ' ')
  say "training $N distractor cartridges into $FIX (currently $before) · GPU0 $N0 · GPU1 $N1 (~6 h)"
  for G in 0 1; do
    CUDA_VISIBLE_DEVICES=$G "$PY" -u scripts/train_joint.py \
      --passages-file $D/corpus.jsonl \
      --conversations-file $D/ss_distract.jsonl \
      --passage-ids "$(paste -sd, - < /tmp/dc_g$G.txt)" \
      --cartridge-dir "$FIX" --cartridge-len 128 \
      --compose-geom gnorm --p-iso 1.0 --epochs 4 --adam8bit \
      --max-answer-tokens 48 --max-question-tokens 96 \
      > out_dc_train_g$G.log 2>&1 &
  done
  wait
  for G in 0 1; do
    grep -q "^Saved" out_dc_train_g$G.log || { say "ABORT: shard $G never saved (see out_dc_train_g$G.log)"; exit 1; }
    grep -h "^epoch" out_dc_train_g$G.log | tail -1
  done
  bad=$(grep -h "^epoch" out_dc_train_g0.log out_dc_train_g1.log | grep -cv "skipped=0  oom=0" || true)
  [ "$bad" -eq 0 ] || { say "ABORT: $bad epoch line(s) report skipped/oom != 0 — do not trust this pool"; exit 1; }
  after=$(find "$FIX" -name 'cartridge_128.pt' | wc -l | tr -d ' ')
  say "pool: $before -> $after cartridges (want +$N)"
  printf '%s\n' \
    "This pool contains 927 GOLD cartridges (375 of them re-distilled on terse self-study," \
    "$(date '+%Y-%m-%d')) PLUS $N DISTRACTOR cartridges added for the coverage-matched substrate" \
    "ablation. The gold-only numbers are NOT reproducible from this directory — they are recorded in" \
    "results/scale500/fix_chain_{terse,verbose}/, run before the distractors existed." > "$FIX/README.txt"
  say "wrote $FIX/README.txt · NEXT: re-run router.py terse+verbose (NO re-index needed)"
fi
