#!/usr/bin/env bash
# Full scientific matrix: Dense 31B × ctx ∈ {1K..128K} × {none, MTP γ=2, MTP γ=4, DFlash dm=4}
# Plus MoE short-ctx extension: 1K..32K × same configs (64K-1M already done in moe-scientific).
# All cells: TQ3/TQ3 KV, --pflash, --temp 0 --seed 0 --ignore-eos, n_predict=64.

cd /home/peppi/Dev/lucebox-hub
export PATH=/usr/local/cuda-13.1/bin:$PATH

LOGDIR=.sisyphus/notes/gemma4-baseline/mtp-gamma/full-matrix
POWDIR=$LOGDIR/power
mkdir -p $LOGDIR $POWDIR

# Models
DENSE=/home/peppi/Dev/lucebox-hub/models/gemma-4-31B-it-Q4_K_M.gguf
DENSE_MTP=/home/peppi/Dev/lucebox-hub/models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q4_K_M.gguf
DENSE_DFLASH=/home/peppi/models/gemma4-31b-dflash/draft-q8_0.gguf
MOE=/home/peppi/models/gemma4-26b-a4b-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
MOE_MTP=/home/peppi/models/gemma4-mtp-26b-a4b/gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf
MOE_DFLASH=/home/peppi/models/gemma4-26b-a4b-dflash/draft-q8_0.gguf

# Prompts by ctx (must fit; choose smallest that's still 'realistic' for the cell)
PROMPTS=.sisyphus/notes/gemma4-baseline/prompts
prompt_for () {
  # Verified token counts: short_chat=26, long_2k=2610, prose_4096=4095,
  # prose_8192=8191, prose_12288=12287, long_code_50k=50003.
  # Prompt+BOS must be ≤ ctx-2 so the binary has decode budget.
  case $1 in
    1024)   echo "$PROMPTS/short_chat.txt" ;;
    4096)   echo "$PROMPTS/long_2k.txt" ;;
    8192)   echo "$PROMPTS/prose_4096.txt" ;;
    16384)  echo "$PROMPTS/prose_8192.txt" ;;
    32768)  echo "$PROMPTS/prose_12288.txt" ;;
    65536)  echo "$PROMPTS/long_code_50k.txt" ;;
    131072) echo "$PROMPTS/long_code_50k.txt" ;;
  esac
}

TS=$LOGDIR/timestamps.csv
: > "$TS"

run () {
  local model=$1 mtp_model=$2 dflash_model=$3 model_tag=$4 mode=$5 ctx=$6
  local prompt; prompt=$(prompt_for "$ctx")
  if [ -z "$prompt" ] || [ ! -f "$prompt" ]; then
    echo "skipping ${model_tag}_${mode}_ctx${ctx}: no prompt"; return
  fi
  local tag="${model_tag}_${mode}_ctx${ctx}"
  local log="$LOGDIR/$tag.log"
  local pow="$POWDIR/$tag.csv"
  echo "=== $tag ($(date +%H:%M:%S)) ==="
  ( while true; do
      printf "%s,%s\n" "$(date +%s.%N)" "$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)"
      sleep 0.1
    done ) > "$pow" 2>/dev/null &
  local POW_PID=$!
  local args=( --model "$model" --ctx-size "$ctx" --n-predict 64 --kv-k tq3_0 --kv-v tq3_0 --temp 0 --seed 0 --ignore-eos --pflash --tokens-file "$prompt" )
  case $mode in
    none)       ;;
    mtp_g2)     args+=( --draft-method mtp --mtp "$mtp_model" --gamma 2 ) ;;
    mtp_g4)     args+=( --draft-method mtp --mtp "$mtp_model" --gamma 4 ) ;;
    dflash_dm4) args+=( --draft-method dflash --draft "$dflash_model" --draft-max 4 ) ;;
  esac
  local t0=$(date +%s.%N)
  timeout 900 ./dflash/build/test_gemma4_dflash "${args[@]}" > "$log" 2>&1
  local rc=$?
  local te=$(date +%s.%N)
  kill $POW_PID 2>/dev/null; wait $POW_PID 2>/dev/null
  echo "$tag rc=$rc t0=$t0 t_end=$te" >> "$TS"
  grep -E '^\[stats\]|chains=|accept_rate|VRAM used|avg_accept' "$log" | tail -3
}

# Dense matrix: 1K..128K (skip 256K+ — VRAM-bound on 24 GB)
for ctx in 1024 4096 8192 16384 32768 65536 131072; do
  for mode in none mtp_g2 mtp_g4 dflash_dm4; do
    run "$DENSE" "$DENSE_MTP" "$DENSE_DFLASH" dense "$mode" "$ctx"
  done
done

# MoE short-ctx extension: 1K..32K (64K-1M already done in moe-scientific)
for ctx in 1024 4096 8192 16384 32768; do
  for mode in none mtp_g2 mtp_g4 dflash_dm4; do
    run "$MOE" "$MOE_MTP" "$MOE_DFLASH" moe "$mode" "$ctx"
  done
done

echo "=== sweep complete $(date +%H:%M:%S) ==="
ls "$LOGDIR"/*.log | wc -l
