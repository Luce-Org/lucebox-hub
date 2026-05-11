#!/usr/bin/env bash
# Phase 4 sweep: γ × ctx on Dense 31B + TQ3/TQ3.  Continue on individual cell failures.

set -o pipefail
ROOT=/home/peppi/Dev/lucebox-hub
BIN=$ROOT/dflash/build/test_gemma4_dflash
TGT=$ROOT/models/gemma-4-31B-it-Q4_K_M.gguf
MTP=$ROOT/models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q4_K_M.gguf
OUT=$ROOT/.sisyphus/notes/gemma4-baseline/mtp-gamma/phase4-b
PROMPTS=$ROOT/.sisyphus/notes/gemma4-baseline/prompts
mkdir -p "$OUT"

ESSAY_4K="Write a brief essay about the impact of speculative decoding on consumer LLM inference. Cover the core idea, why it works, and practical limits."

# Returns: "--prompt|TEXT" or "--tokens-file|PATH"
prompt_args () {
  case $1 in
    4096)  echo "--prompt|$ESSAY_4K" ;;
    16384) echo "--tokens-file|$PROMPTS/prose_12288.txt" ;;
    65536) echo "--tokens-file|$PROMPTS/long_code_50k.txt" ;;
  esac
}

run_one () {
  local label=$1 gamma=$2 ctx=$3
  local pair flag value tag log
  pair=$(prompt_args "$ctx")
  flag=${pair%|*}
  value=${pair#*|}
  tag="${label}_g${gamma}_ctx${ctx}"
  log="$OUT/$tag.log"
  echo "=== $tag ($(date +%H:%M:%S)) ==="
  local args=( --model "$TGT" --ctx-size "$ctx" --n-predict 64 --kv-k tq3_0 --kv-v tq3_0 --temp 0 --ignore-eos )
  if [[ $label == mtp ]]; then
    args+=( --draft-method mtp --mtp "$MTP" --gamma "$gamma" )
  fi
  args+=( "$flag" "$value" )
  timeout 600 "$BIN" "${args[@]}" > "$log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "  rc=$rc — last line: $(tail -1 "$log")"
  fi
  grep -E '^\[stats\]|chains=|accept_rate' "$log" | tail -3
}

# Phase 4 matrix
for c in 4096 16384 65536; do run_one none 0 "$c"; done
for c in 4096 16384 65536; do
  for g in 1 2 4 8; do
    run_one mtp "$g" "$c"
  done
done

echo "=== sweep complete $(date +%H:%M:%S) ==="
ls "$OUT"/*.log | wc -l
