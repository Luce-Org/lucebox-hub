#!/usr/bin/env bash
# (1) Verify dense 31B + Q8/Q8 at 64k actually OOMs or not.
# (2) Sweep MoE 26B + dflash + Q8/Q8 + dm=4 across context sizes from 16k upward.
set -e
cd /home/peppi/Dev/lucebox-hub
export PATH=/usr/local/cuda-13.1/bin:$PATH

LOGDIR=.sisyphus/notes/gemma4-baseline/scaling
mkdir -p $LOGDIR

DENSE=models/gemma-4-31B-it-Q4_K_M.gguf
MOE=/home/peppi/models/gemma4-26b-a4b-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
MOE_DFLASH=/home/peppi/models/gemma4-26b-a4b-dflash/draft-q8_0.gguf

PROMPT_SHORT=.sisyphus/notes/gemma4-baseline/prompts/long_open.txt   # 40 toks
PROMPT_2K=.sisyphus/notes/gemma4-baseline/prompts/long_2k.txt        # 2611 toks
PROMPT_50K=.sisyphus/notes/gemma4-baseline/prompts/long_50k.txt      # 49904 toks

echo "# Scaling matrix — $(date -Iseconds)" > $LOGDIR/SUMMARY.md

run() {
  local tag=$1; local logfile=$LOGDIR/${tag}.log
  shift
  echo "=== $tag starting at $(date +%H:%M:%S) ===" | tee -a $LOGDIR/SUMMARY.md
  ./dflash/build/test_gemma4_dflash "$@" \
    --n-predict 256 --temp 0 --seed 0 --ignore-eos \
    > $logfile 2>&1 || true
  local rc=$?
  echo "$tag rc=$rc" | tee -a $LOGDIR/SUMMARY.md
}

# (1) Dense 31B + Q8/Q8 at 64k — test the assumption it OOMs
run D1_dense_q8_q8_64k \
  --model $DENSE \
  --tokens-file $PROMPT_50K \
  --kv-k q8_0 --kv-v q8_0 \
  --ctx-size 65536 --pflash \
  --draft-method none

# (2) MoE 26B sweep — Q8/Q8, dflash drafter, dm=4 (user-recommended for MoE)
# Steps: 16k, 32k, 64k, 128k, 256k. Use long_50k where it fits, smaller prompt where it doesn't.
for ctx in 16384 32768 65536 131072 262144; do
  if [ $ctx -ge 65536 ]; then PROMPT=$PROMPT_50K; else PROMPT=$PROMPT_2K; fi
  run M_moe_dflash_q8q8_${ctx} \
    --model $MOE \
    --draft $MOE_DFLASH \
    --draft-method dflash --draft-max 4 \
    --tokens-file $PROMPT \
    --kv-k q8_0 --kv-v q8_0 \
    --ctx-size $ctx --pflash
done

# Compact summary
echo "" >> $LOGDIR/SUMMARY.md
echo "## Per-cell stats" >> $LOGDIR/SUMMARY.md
for log in $LOGDIR/*.log; do
  tag=$(basename $log .log)
  echo "" >> $LOGDIR/SUMMARY.md
  echo "### $tag" >> $LOGDIR/SUMMARY.md
  echo '```' >> $LOGDIR/SUMMARY.md
  grep -E "context_used|kv types|prefill.*tokens in|tok/s=|VRAM used|^\[mtp\] steps|^\[spec\]|GGML_ABORT|out of memory|cudaMalloc|^\[draft\] KV" $log >> $LOGDIR/SUMMARY.md
  echo '```' >> $LOGDIR/SUMMARY.md
done

echo "" | tee -a $LOGDIR/SUMMARY.md
echo "DONE" | tee -a $LOGDIR/SUMMARY.md
