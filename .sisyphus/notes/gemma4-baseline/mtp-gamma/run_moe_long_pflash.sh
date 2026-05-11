#!/usr/bin/env bash
# MoE 26B-A4B at 512K + 1M with pflash; no-MTP and DFlash dm=4

set -o pipefail
ROOT=/home/peppi/Dev/lucebox-hub
BIN=$ROOT/dflash/build/test_gemma4_dflash
TGT=/home/peppi/models/gemma4-26b-a4b-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
DRAFT=/home/peppi/models/gemma4-26b-a4b-dflash/draft-q8_0.gguf
OUT=$ROOT/.sisyphus/notes/gemma4-baseline/mtp-gamma/moe-long-pflash
PROMPT=$ROOT/.sisyphus/notes/gemma4-baseline/prompts/long_code_50k.txt
mkdir -p "$OUT"

run () {
  local label=$1 ctx=$2
  local tag="${label}_ctx${ctx}"
  local log="$OUT/$tag.log"
  echo "=== $tag ($(date +%H:%M:%S)) ==="
  local args=( --model "$TGT" --ctx-size "$ctx" --n-predict 64 --kv-k tq3_0 --kv-v tq3_0 --temp 0 --ignore-eos --pflash --tokens-file "$PROMPT" )
  [[ $label == dflash_dm4 ]] && args+=( --draft-method dflash --draft "$DRAFT" --draft-max 4 )
  timeout 900 "$BIN" "${args[@]}" > "$log" 2>&1 || echo "  rc=$? — $(tail -1 "$log")"
  grep -E '^\[stats\]|chains=|accept_rate|\[mem\]' "$log" | tail -4
}

for c in 524288 1048576; do
  run none       "$c"
  run dflash_dm4 "$c"
done
echo "=== sweep complete $(date +%H:%M:%S) ==="
