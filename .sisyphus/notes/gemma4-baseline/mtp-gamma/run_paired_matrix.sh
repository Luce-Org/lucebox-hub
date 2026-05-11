#!/usr/bin/env bash
# Paired-prompt scientific matrix: Dense + MoE × ctx × {none, MTP γ=2, MTP γ=4, DFlash dm=4}
#                                × {code, prose} prompts at each ctx.
# All cells: TQ3/TQ3 KV, --pflash, --temp 0 --seed 0 --ignore-eos, n_predict=64,
#            per-cell GPU power telemetry → joules.

cd /home/peppi/Dev/lucebox-hub
export PATH=/usr/local/cuda-13.1/bin:$PATH

LOGDIR=.sisyphus/notes/gemma4-baseline/mtp-gamma/paired-matrix
POWDIR=$LOGDIR/power
mkdir -p $LOGDIR $POWDIR

DENSE=/home/peppi/Dev/lucebox-hub/models/gemma-4-31B-it-Q4_K_M.gguf
DENSE_MTP=/home/peppi/Dev/lucebox-hub/models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q4_K_M.gguf
DENSE_DFLASH=/home/peppi/models/gemma4-31b-dflash/draft-q8_0.gguf
MOE=/home/peppi/models/gemma4-26b-a4b-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
MOE_MTP=/home/peppi/models/gemma4-mtp-26b-a4b/gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf
MOE_DFLASH=/home/peppi/models/gemma4-26b-a4b-dflash/draft-q8_0.gguf

P=.sisyphus/notes/gemma4-baseline/prompts
prompt_for () {
  # $1 = ctx, $2 = code|prose
  case "$1:$2" in
    1024:code)    echo "$P/humaneval_2.txt" ;;
    1024:prose)   echo "$P/long_open.txt" ;;
    4096:code)    echo "$P/code_2k.txt" ;;
    4096:prose)   echo "$P/long_2k.txt" ;;
    8192:code)    echo "$P/code_4k.txt" ;;
    8192:prose)   echo "$P/prose_4096.txt" ;;
    16384:code)   echo "$P/code_8k.txt" ;;
    16384:prose)  echo "$P/prose_8192.txt" ;;
    32768:code)   echo "$P/code_12k.txt" ;;
    32768:prose)  echo "$P/prose_12288.txt" ;;
    65536:code|131072:code|262144:code|524288:code|1048576:code) echo "$P/long_code_50k.txt" ;;
    65536:prose|131072:prose|262144:prose|524288:prose|1048576:prose) echo "$P/long_50k.txt" ;;
  esac
}

TS=$LOGDIR/timestamps.csv
: > "$TS"

run () {
  local model_tag=$1 model=$2 mtp=$3 dflash=$4 mode=$5 ctx=$6 distro=$7
  local prompt; prompt=$(prompt_for "$ctx" "$distro")
  [ -z "$prompt" ] || [ ! -f "$prompt" ] && { echo "skip ${model_tag}_${mode}_${distro}_ctx${ctx}: no prompt"; return; }
  local tag="${model_tag}_${mode}_${distro}_ctx${ctx}"
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
    mtp_g2)     args+=( --draft-method mtp --mtp "$mtp" --gamma 2 ) ;;
    mtp_g4)     args+=( --draft-method mtp --mtp "$mtp" --gamma 4 ) ;;
    dflash_dm4) args+=( --draft-method dflash --draft "$dflash" --draft-max 4 ) ;;
  esac
  local t0=$(date +%s.%N)
  timeout 900 ./dflash/build/test_gemma4_dflash "${args[@]}" > "$log" 2>&1
  local rc=$?
  local te=$(date +%s.%N)
  kill $POW_PID 2>/dev/null; wait $POW_PID 2>/dev/null
  echo "$tag rc=$rc t0=$t0 t_end=$te" >> "$TS"
  grep -E '^\[stats\]|chains=|accept_rate|VRAM used|avg_accept' "$log" | tail -3
}

# Dense 1K..128K × 4 modes × 2 prompts = 56 cells
for ctx in 1024 4096 8192 16384 32768 65536 131072; do
  for mode in none mtp_g2 mtp_g4 dflash_dm4; do
    for distro in code prose; do
      run dense "$DENSE" "$DENSE_MTP" "$DENSE_DFLASH" "$mode" "$ctx" "$distro"
    done
  done
done

# MoE short 1K..32K × 4 × 2 = 40 cells
for ctx in 1024 4096 8192 16384 32768; do
  for mode in none mtp_g2 mtp_g4 dflash_dm4; do
    for distro in code prose; do
      run moe "$MOE" "$MOE_MTP" "$MOE_DFLASH" "$mode" "$ctx" "$distro"
    done
  done
done

# MoE long 64K..1M × 4 × prose only (code already in moe-scientific dir) = 20 cells
for ctx in 65536 131072 262144 524288 1048576; do
  for mode in none mtp_g2 mtp_g4 dflash_dm4; do
    run moe "$MOE" "$MOE_MTP" "$MOE_DFLASH" "$mode" "$ctx" prose
  done
done

echo "=== sweep complete $(date +%H:%M:%S) ==="
ls "$LOGDIR"/*.log | wc -l
