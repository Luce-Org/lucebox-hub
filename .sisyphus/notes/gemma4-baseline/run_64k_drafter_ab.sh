#!/usr/bin/env bash
# 64k context, TQ3 KV, pFlash on, dense 31B.
# Compare drafters: target-only vs MTP vs dflash.
set -e
cd /home/peppi/Dev/lucebox-hub
export PATH=/usr/local/cuda-13.1/bin:$PATH

LOGDIR=.sisyphus/notes/gemma4-baseline/matrix-64k
mkdir -p $LOGDIR

MODEL=models/gemma-4-31B-it-Q4_K_M.gguf
MTP=models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q8_0.gguf
DFLASH_DIR=models/draft-gemma4-31b
PROMPT=.sisyphus/notes/gemma4-baseline/prompts/long_50k.txt

CTX=65536
NPREDICT=256

run_cell() {
  local tag=$1; shift
  local logfile=$LOGDIR/${tag}.log
  echo "=== ${tag} ===" | tee -a $LOGDIR/SUMMARY.md
  ./dflash/build/test_gemma4_dflash \
    --model $MODEL \
    --tokens-file $PROMPT \
    --kv-k tq3_0 --kv-v tq3_0 \
    --ctx-size $CTX --pflash \
    --n-predict $NPREDICT --temp 0 --seed 0 --ignore-eos \
    "$@" \
    > $logfile 2>&1
  local rc=$?
  echo "${tag} rc=$rc" | tee -a $LOGDIR/SUMMARY.md
  return $rc
}

echo "# 64k drafter A/B with TQ3 + pFlash (dense 31B) — $(date -Iseconds)" > $LOGDIR/SUMMARY.md
echo "Prompt: long_50k.txt (~50k tokens), ctx=$CTX, n_predict=$NPREDICT" >> $LOGDIR/SUMMARY.md
echo "" >> $LOGDIR/SUMMARY.md

# T1: target-only (baseline)
run_cell T1_none --draft-method none || echo "T1 failed but continuing"

# T2: MTP drafter
run_cell T2_mtp --draft-method mtp --mtp $MTP || echo "T2 failed but continuing"

# T3: dflash drafter
run_cell T3_dflash --draft-method dflash --draft $DFLASH_DIR || echo "T3 failed but continuing"

# Stats extraction
echo "" >> $LOGDIR/SUMMARY.md
echo "## Per-cell stats" >> $LOGDIR/SUMMARY.md
for cell in T1_none T2_mtp T3_dflash; do
  log=$LOGDIR/${cell}.log
  [ -f $log ] || continue
  echo "" >> $LOGDIR/SUMMARY.md
  echo "### ${cell}" >> $LOGDIR/SUMMARY.md
  echo '```' >> $LOGDIR/SUMMARY.md
  grep -E "kv types|narrow asymmetric|pflash|prefill.*tokens in|context_used|tok/s=|VRAM used|^\[mtp\] steps|accept_rate|GGML_ABORT|fatal" $log 2>&1 | head -25 >> $LOGDIR/SUMMARY.md
  echo '```' >> $LOGDIR/SUMMARY.md
done

# Decoded text comparison (first 80 generated tokens for each)
echo "" >> $LOGDIR/SUMMARY.md
echo "## First 80 generated tokens (decoded)" >> $LOGDIR/SUMMARY.md
python3 - <<'PY' >> $LOGDIR/SUMMARY.md
import re, os
from transformers import AutoTokenizer
LOGDIR = ".sisyphus/notes/gemma4-baseline/matrix-64k"
t = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

for cell in ("T1_none", "T2_mtp", "T3_dflash"):
    p = f"{LOGDIR}/{cell}.log"
    if not os.path.exists(p):
        print(f"\n### {cell}: NO LOG"); continue
    with open(p) as f: log = f.read()
    # Slice between [prefill] ... ms and [stats]/[mtp] block
    if "[prefill]" not in log:
        print(f"\n### {cell}: no [prefill] marker"); continue
    body = log.split("[prefill]", 2)[-1]
    body = body.split("[stats]")[0]
    body = re.sub(r'\[mtp-step \d+\] accept_rate=[\d.]+', '', body)
    body = re.sub(r'ggml_backend_cuda_graph_compute:.*?\n', '', body)
    body = re.sub(r'\[mtp-dbg\][^\n]*\n', '', body)
    nums = re.findall(r'(?<![a-zA-Z=:])\b(\d+)\b', body)
    ids = [int(x) for x in nums if int(x) < 262144]
    # Skip tokens that are obviously prefill metadata leaks (small numbers in first ~10)
    # Heuristic: real generation starts after first significant gap; for now just take from index where we see >1000 sequence of changes
    print(f"\n### {cell}")
    print(f"raw extracted (first 80): {ids[:80]}")
    print(f"decoded (first 80): {repr(t.decode(ids[:80], skip_special_tokens=False))}")
PY

echo "" >> $LOGDIR/SUMMARY.md
echo "DONE" >> $LOGDIR/SUMMARY.md
echo "" | tee -a $LOGDIR/SUMMARY.md
echo "All cells complete. See $LOGDIR/SUMMARY.md"
