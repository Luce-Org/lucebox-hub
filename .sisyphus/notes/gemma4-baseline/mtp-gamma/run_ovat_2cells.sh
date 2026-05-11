#!/usr/bin/env bash
# OVAT at two more reference cells: MoE 1M code, Dense 64K code.
# Same 10-cell design as ovat-moe-64k-code.

cd /home/peppi/Dev/lucebox-hub
export PATH=/usr/local/cuda-13.1/bin:$PATH

MOE=/home/peppi/models/gemma4-26b-a4b-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
MOE_MTP=/home/peppi/models/gemma4-mtp-26b-a4b/gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf
MOE_DFLASH=/home/peppi/models/gemma4-26b-a4b-dflash/draft-q8_0.gguf

DENSE=/home/peppi/Dev/lucebox-hub/models/gemma-4-31B-it-Q4_K_M.gguf
DENSE_MTP=/home/peppi/Dev/lucebox-hub/models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q4_K_M.gguf
DENSE_DFLASH=/home/peppi/models/gemma4-31b-dflash/draft-q8_0.gguf

PROMPT=.sisyphus/notes/gemma4-baseline/prompts/long_code_50k.txt

# Args: $1 logdir tag $2 model $3 mtp_model $4 dflash_model $5 ctx $6 cell_tag $7 kv $8 pflash $9 drafter_spec
run_cell () {
  local LOGDIR=$1 model=$2 mtp_model=$3 dflash_model=$4 ctx=$5
  local tag=$6 kv=$7 pflash=$8 drafter=$9
  local POWDIR=$LOGDIR/power
  mkdir -p $POWDIR
  local log=$LOGDIR/${tag}.log
  local pow=$POWDIR/${tag}.csv
  echo "=== ${tag} ctx=${ctx} ($(date +%H:%M:%S)) ==="
  ( while true; do
      printf "%s,%s\n" "$(date +%s.%N)" "$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)"
      sleep 0.1
    done ) > "$pow" 2>/dev/null &
  local POW_PID=$!
  local args=( --model "$model" --ctx-size "$ctx" --n-predict 64 --temp 0 --seed 0 --ignore-eos --tokens-file "$PROMPT" )
  case $kv in
    q8)  args+=( --kv-k q8_0  --kv-v q8_0  ) ;;
    tq3) args+=( --kv-k tq3_0 --kv-v tq3_0 ) ;;
  esac
  [[ $pflash == on ]] && args+=( --pflash )
  case $drafter in
    none) ;;
    dflash:dm*) dm=${drafter##*:dm}; args+=( --draft-method dflash --draft "$dflash_model" --draft-max "$dm" ) ;;
    mtp:g*)     gm=${drafter##*:g};  args+=( --draft-method mtp --mtp "$mtp_model" --gamma "$gm" ) ;;
  esac
  local t0=$(date +%s.%N)
  timeout 900 ./dflash/build/test_gemma4_dflash "${args[@]}" > "$log" 2>&1
  local rc=$?
  local te=$(date +%s.%N)
  kill $POW_PID 2>/dev/null; wait $POW_PID 2>/dev/null
  echo "$tag rc=$rc t0=$t0 t_end=$te" >> $LOGDIR/timestamps.csv
  grep -E '^\[stats\]|\[mem\]|accept_rate|avg_accept|error|fatal' "$log" | tail -3
}

run_ovat_suite () {
  local LOGDIR=$1 model=$2 mtp_model=$3 dflash_model=$4 ctx=$5
  mkdir -p $LOGDIR
  : > $LOGDIR/timestamps.csv
  run_cell $LOGDIR $model $mtp_model $dflash_model $ctx 00_naive_q8     q8  off none
  run_cell $LOGDIR $model $mtp_model $dflash_model $ctx 01_tq3          tq3 off none
  run_cell $LOGDIR $model $mtp_model $dflash_model $ctx 02_q8_pflash    q8  on  none
  run_cell $LOGDIR $model $mtp_model $dflash_model $ctx 03_tq3_pflash   tq3 on  none
  run_cell $LOGDIR $model $mtp_model $dflash_model $ctx 04_tq3_pf_dfl4  tq3 on  dflash:dm4
  run_cell $LOGDIR $model $mtp_model $dflash_model $ctx 05_tq3_pf_dfl8  tq3 on  dflash:dm8
  run_cell $LOGDIR $model $mtp_model $dflash_model $ctx 06_tq3_pf_dfl16 tq3 on  dflash:dm16
  run_cell $LOGDIR $model $mtp_model $dflash_model $ctx 07_tq3_pf_mtp1  tq3 on  mtp:g1
  run_cell $LOGDIR $model $mtp_model $dflash_model $ctx 08_tq3_pf_mtp2  tq3 on  mtp:g2
  run_cell $LOGDIR $model $mtp_model $dflash_model $ctx 09_tq3_pf_mtp4  tq3 on  mtp:g4
}

echo "### Suite A: MoE 26B-A4B @ 1M code ($(date +%H:%M:%S)) ###"
run_ovat_suite .sisyphus/notes/gemma4-baseline/mtp-gamma/ovat-moe-1M-code $MOE $MOE_MTP $MOE_DFLASH 1048576

echo "### Suite B: Dense 31B @ 64K code ($(date +%H:%M:%S)) ###"
run_ovat_suite .sisyphus/notes/gemma4-baseline/mtp-gamma/ovat-dense-64k-code $DENSE $DENSE_MTP $DENSE_DFLASH 65536

echo "=== both sweeps complete $(date +%H:%M:%S) ==="
