#!/usr/bin/env bash
set -o pipefail
ROOT=/home/peppi/Dev/lucebox-hub
BIN=$ROOT/dflash/build/test_gemma4_dflash
TGT=$ROOT/models/gemma-4-31B-it-Q4_K_M.gguf
MTP=$ROOT/models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q4_K_M.gguf
OUT=$ROOT/.sisyphus/notes/gemma4-baseline/mtp-gamma/phase4-b-long
PROMPT=$ROOT/.sisyphus/notes/gemma4-baseline/prompts/long_code_50k.txt
mkdir -p "$OUT"

run () {
  local label=$1 gamma=$2
  local tag="${label}_g${gamma}_ctx131072"
  local log="$OUT/$tag.log"
  echo "=== $tag ($(date +%H:%M:%S)) ==="
  local args=( --model "$TGT" --ctx-size 131072 --n-predict 64 --kv-k tq3_0 --kv-v tq3_0 --temp 0 --ignore-eos --tokens-file "$PROMPT" )
  [[ $label == mtp ]] && args+=( --draft-method mtp --mtp "$MTP" --gamma "$gamma" )
  timeout 600 "$BIN" "${args[@]}" > "$log" 2>&1 || echo "  rc=$? — $(tail -1 "$log")"
  grep -E '^\[stats\]|chains=|accept_rate' "$log" | tail -3
}

# 131072 only (256K and 1M take >15 min prefill on Dense 31B + TQ3, beyond scope)
# no-MTP already ran at 6.24 tok/s; rerun to confirm
run none 0
for g in 1 2 4; do run mtp "$g"; done
echo "=== sweep complete $(date +%H:%M:%S) ==="
