#!/usr/bin/env bash
# Phase 4-B-long: extend γ=2/γ=4 to 128K, 256K, 1M

set -o pipefail
ROOT=/home/peppi/Dev/lucebox-hub
BIN=$ROOT/dflash/build/test_gemma4_dflash
TGT=$ROOT/models/gemma-4-31B-it-Q4_K_M.gguf
MTP=$ROOT/models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q4_K_M.gguf
OUT=$ROOT/.sisyphus/notes/gemma4-baseline/mtp-gamma/phase4-b-long
PROMPT=$ROOT/.sisyphus/notes/gemma4-baseline/prompts/long_code_50k.txt
mkdir -p "$OUT"

run_one () {
  local label=$1 gamma=$2 ctx=$3
  local tag="${label}_g${gamma}_ctx${ctx}"
  local log="$OUT/$tag.log"
  echo "=== $tag ($(date +%H:%M:%S)) ==="
  local args=( --model "$TGT" --ctx-size "$ctx" --n-predict 64 --kv-k tq3_0 --kv-v tq3_0 --temp 0 --ignore-eos --tokens-file "$PROMPT" )
  if [[ $label == mtp ]]; then
    args+=( --draft-method mtp --mtp "$MTP" --gamma "$gamma" )
  fi
  timeout 900 "$BIN" "${args[@]}" > "$log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "  rc=$rc — last line: $(tail -1 "$log")"
  fi
  grep -E '^\[stats\]|chains=|accept_rate|^\[mem\]' "$log" | tail -4
}

# No-MTP + γ=1/2/4 at each long context.  Skip γ=8 (consistently underperforms).
for c in 131072 262144 1048576; do run_one none 0 "$c"; done
for c in 131072 262144 1048576; do
  for g in 1 2 4; do
    run_one mtp "$g" "$c"
  done
done

echo "=== sweep complete $(date +%H:%M:%S) ==="
ls "$OUT"/*.log | wc -l
