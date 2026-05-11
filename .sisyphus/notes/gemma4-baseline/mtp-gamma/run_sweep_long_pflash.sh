#!/usr/bin/env bash
# Phase 4-B-long with pflash: 128K/256K/1M sweep on Dense 31B + TQ3/TQ3
# Adds --pflash to every cell (was the bottleneck for the prior 256K hang).

set -o pipefail
ROOT=/home/peppi/Dev/lucebox-hub
BIN=$ROOT/dflash/build/test_gemma4_dflash
TGT=$ROOT/models/gemma-4-31B-it-Q4_K_M.gguf
MTP=$ROOT/models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q4_K_M.gguf
OUT=$ROOT/.sisyphus/notes/gemma4-baseline/mtp-gamma/phase4-b-long-pflash
PROMPT=$ROOT/.sisyphus/notes/gemma4-baseline/prompts/long_code_50k.txt
mkdir -p "$OUT"

run () {
  local label=$1 gamma=$2 ctx=$3
  local tag="${label}_g${gamma}_ctx${ctx}"
  local log="$OUT/$tag.log"
  echo "=== $tag ($(date +%H:%M:%S)) ==="
  local args=( --model "$TGT" --ctx-size "$ctx" --n-predict 64 --kv-k tq3_0 --kv-v tq3_0 --temp 0 --ignore-eos --pflash --tokens-file "$PROMPT" )
  [[ $label == mtp ]] && args+=( --draft-method mtp --mtp "$MTP" --gamma "$gamma" )
  timeout 900 "$BIN" "${args[@]}" > "$log" 2>&1 || echo "  rc=$? — $(tail -1 "$log")"
  grep -E '^\[stats\]|chains=|accept_rate|\[mem\]' "$log" | tail -4
}

# 128K first (fastest), then 256K, then 1M.  Skip γ=8 (consistently underperforms).
for c in 131072 262144 1048576; do run none 0 "$c"; done
for c in 131072 262144 1048576; do
  for g in 1 2 4; do run mtp "$g" "$c"; done
done
echo "=== sweep complete $(date +%H:%M:%S) ==="
ls "$OUT"/*.log | wc -l
