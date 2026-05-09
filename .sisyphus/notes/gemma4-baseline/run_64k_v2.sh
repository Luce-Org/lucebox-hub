#!/usr/bin/env bash
# 64k matrix v2 — all 3 fixes in (SWA mask + TQ3 dispatcher + head_dim=512 mask).
# Dense 31B + TQ3/TQ3 + pflash + ctx 65536 + 50k prompt.
set -e
cd /home/peppi/Dev/lucebox-hub
export PATH=/usr/local/cuda-13.1/bin:$PATH

LOGDIR=.sisyphus/notes/gemma4-baseline/matrix-64k-v2
mkdir -p $LOGDIR

MODEL=models/gemma-4-31B-it-Q4_K_M.gguf
MTP=models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q8_0.gguf
DFLASH_GGUF=dflash/models/draft-gemma4-31b/draft-q8_0.gguf
PROMPT=.sisyphus/notes/gemma4-baseline/prompts/long_50k.txt

CTX=65536
NPREDICT=256

run_cell() {
  local tag=$1; shift
  echo "=== ${tag} starting at $(date +%H:%M:%S) ===" | tee -a $LOGDIR/SUMMARY.md
  ./dflash/build/test_gemma4_dflash \
    --model $MODEL \
    --tokens-file $PROMPT \
    --kv-k tq3_0 --kv-v tq3_0 \
    --ctx-size $CTX --pflash \
    --n-predict $NPREDICT --temp 0 --seed 0 --ignore-eos \
    "$@" \
    > $LOGDIR/${tag}.log 2>&1 || true
  local rc=$?
  echo "${tag} rc=$rc" | tee -a $LOGDIR/SUMMARY.md
}

echo "# Matrix v2 at 64k — all fixes in. $(date -Iseconds)" > $LOGDIR/SUMMARY.md
echo "" >> $LOGDIR/SUMMARY.md

run_cell V1_none --draft-method none
run_cell V2_mtp  --draft-method mtp --mtp $MTP
run_cell V3_dflash_dm8 --draft-method dflash --draft $DFLASH_GGUF --draft-max 8

echo "" >> $LOGDIR/SUMMARY.md
echo "## Per-cell stats" >> $LOGDIR/SUMMARY.md
for cell in V1_none V2_mtp V3_dflash_dm8; do
  log=$LOGDIR/${cell}.log
  [ -f $log ] || continue
  echo "" >> $LOGDIR/SUMMARY.md
  echo "### ${cell}" >> $LOGDIR/SUMMARY.md
  echo '```' >> $LOGDIR/SUMMARY.md
  grep -E "kv types|narrow asymmetric|^\[draft\] KV|prefill.*tokens in|context_used|tok/s=|VRAM used|^\[mtp\] steps|^\[spec\]|GGML_ABORT" $log >> $LOGDIR/SUMMARY.md
  echo '```' >> $LOGDIR/SUMMARY.md
done

echo "" >> $LOGDIR/SUMMARY.md
echo "## Decoded text comparison (first 80 generated tokens)" >> $LOGDIR/SUMMARY.md
python3 - <<'PY' >> $LOGDIR/SUMMARY.md
import re, os
from transformers import AutoTokenizer
LOGDIR = ".sisyphus/notes/gemma4-baseline/matrix-64k-v2"
t = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
for cell in ("V1_none", "V2_mtp", "V3_dflash_dm8"):
    p = f"{LOGDIR}/{cell}.log"
    if not os.path.exists(p): print(f"\n### {cell}: NO LOG"); continue
    with open(p) as f: log = f.read()
    if "[prefill]" not in log: print(f"\n### {cell}: no [prefill] marker"); continue
    body = log.split("[prefill]", 2)[-1].split("[stats]")[0]
    body = re.sub(r'\[mtp-step \d+\] accept_rate=[\d.]+', '', body)
    body = re.sub(r'\[step \d+\] accept=\d+/\d+ avg=[\d.]+', '', body)
    body = re.sub(r'ggml_backend[^\n]*\n', '', body)
    body = re.sub(r'\[(mtp-dbg|draft)[^\]]*\][^\n]*\n', '', body)
    nums = re.findall(r'(?<![a-zA-Z=:])\b(\d+)\b', body)
    ids = [int(x) for x in nums if int(x) < 262144]
    print(f"\n### {cell}")
    print(f"first_80_decoded: {repr(t.decode(ids[:80], skip_special_tokens=False))}")
PY

echo "" | tee -a $LOGDIR/SUMMARY.md
echo "DONE" | tee -a $LOGDIR/SUMMARY.md
