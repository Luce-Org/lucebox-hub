#!/usr/bin/env bash
# Worst-case retest: MoE 26B + dflash + Q8/Q8 + pflash at 64k AND 256k,
# WITH a 50k-token CODE prompt (long_code_50k.txt) instead of the Shakespeare one.
# Plus draft-max sweep ∈ {1, 2, 4, 8} on the 64k cell to find the optimum at long ctx with code.
set -e
cd /home/peppi/Dev/lucebox-hub
export PATH=/usr/local/cuda-13.1/bin:$PATH

LOGDIR=.sisyphus/notes/gemma4-baseline/dm-sweep
mkdir -p $LOGDIR

MOE=/home/peppi/models/gemma4-26b-a4b-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
MOE_DFLASH=/home/peppi/models/gemma4-26b-a4b-dflash/draft-q8_0.gguf
PROMPT_CODE=.sisyphus/notes/gemma4-baseline/prompts/long_code_50k.txt

echo "# DM sweep on long-code prompt — $(date -Iseconds)" > $LOGDIR/SUMMARY.md

run() {
  local tag=$1; local ctx=$2; local dm=$3
  echo "=== ${tag} (ctx=${ctx} dm=${dm}) starting at $(date +%H:%M:%S) ===" | tee -a $LOGDIR/SUMMARY.md
  ./dflash/build/test_gemma4_dflash \
    --model $MOE \
    --draft $MOE_DFLASH \
    --draft-method dflash --draft-max $dm \
    --tokens-file $PROMPT_CODE \
    --kv-k q8_0 --kv-v q8_0 \
    --ctx-size $ctx --pflash \
    --n-predict 256 --temp 0 --seed 0 --ignore-eos \
    > $LOGDIR/${tag}.log 2>&1 || true
  echo "${tag} rc=$?" | tee -a $LOGDIR/SUMMARY.md
}

# 64k: dm sweep on code prompt
for dm in 1 2 4 8; do
  run code64k_dm${dm} 65536 $dm
done

# 256k with the best/typical dm to verify at the high end
run code256k_dm4 262144 4
run code256k_dm8 262144 8

# 4k with code (sanity baseline — should hit the best AL)
run code4k_dm4 4096 4
run code4k_dm8 4096 8

echo "" >> $LOGDIR/SUMMARY.md
echo "## Per-cell stats" >> $LOGDIR/SUMMARY.md
for log in $LOGDIR/*.log; do
  tag=$(basename $log .log)
  echo "" >> $LOGDIR/SUMMARY.md
  echo "### $tag" >> $LOGDIR/SUMMARY.md
  echo '```' >> $LOGDIR/SUMMARY.md
  grep -E "context_used|prefill.*tokens in|tok/s=|VRAM used|^\[spec\]|GGML_ABORT|^\[draft\] KV" $log >> $LOGDIR/SUMMARY.md
  echo '```' >> $LOGDIR/SUMMARY.md
done

echo "" | tee -a $LOGDIR/SUMMARY.md
echo "DONE" | tee -a $LOGDIR/SUMMARY.md
