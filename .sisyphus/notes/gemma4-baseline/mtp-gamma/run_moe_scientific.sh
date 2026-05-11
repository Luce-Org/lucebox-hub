#!/usr/bin/env bash
# Scientific sweep: MoE 26B-A4B × {none, MTP γ=2, DFlash dm=4} × {64K, 128K, 256K, 512K, 1M}
# All cells: TQ3/TQ3 KV, --pflash, --temp 0 --ignore-eos, n_predict=64
# Captures per-cell GPU power profile → total/prefill/decode energy + J/tok.

cd /home/peppi/Dev/lucebox-hub
export PATH=/usr/local/cuda-13.1/bin:$PATH

LOGDIR=.sisyphus/notes/gemma4-baseline/mtp-gamma/moe-scientific
POWLOG=$LOGDIR/power
mkdir -p $LOGDIR $POWLOG

MOE=/home/peppi/models/gemma4-26b-a4b-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
MOE_MTP=/home/peppi/models/gemma4-mtp-26b-a4b/gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf
MOE_DFLASH=/home/peppi/models/gemma4-26b-a4b-dflash/draft-q8_0.gguf
PROMPT=.sisyphus/notes/gemma4-baseline/prompts/long_code_50k.txt
TS_CSV=$LOGDIR/timestamps.csv
: > "$TS_CSV"

echo "# MoE scientific bench — $(date -Iseconds)" > $LOGDIR/SUMMARY.md
echo "MoE 26B-A4B + TQ3/TQ3 + pflash, 50K code prompt, n_predict=64, --temp 0 --ignore-eos --seed 0" >> $LOGDIR/SUMMARY.md

run_cell() {
  local mode=$1 ctx=$2 tag=$3
  local logfile=$LOGDIR/${tag}.log
  local powfile=$POWLOG/${tag}.csv

  ( while true; do
      printf "%s,%s\n" "$(date +%s.%N)" "$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)"
      sleep 0.1
    done ) > "$powfile" 2>/dev/null &
  local POW_PID=$!

  local t0=$(date +%s.%N)
  local args=( --model "$MOE" --ctx-size "$ctx" --n-predict 64 --kv-k tq3_0 --kv-v tq3_0 --temp 0 --seed 0 --ignore-eos --pflash --tokens-file "$PROMPT" )
  case $mode in
    none)
      ;;
    mtp_g2)
      args+=( --draft-method mtp --mtp "$MOE_MTP" --gamma 2 )
      ;;
    mtp_g4)
      args+=( --draft-method mtp --mtp "$MOE_MTP" --gamma 4 )
      ;;
    dflash_dm4)
      args+=( --draft-method dflash --draft "$MOE_DFLASH" --draft-max 4 )
      ;;
  esac
  timeout 900 ./dflash/build/test_gemma4_dflash "${args[@]}" > "$logfile" 2>&1
  local rc=$?
  local t_end=$(date +%s.%N)
  kill $POW_PID 2>/dev/null; wait $POW_PID 2>/dev/null
  echo "$tag rc=$rc t0=$t0 t_end=$t_end" >> "$TS_CSV"
  echo "=== $tag rc=$rc ($(date +%H:%M:%S)) ==="
  grep -E '^\[stats\]|chains=|accept_rate|\[mem\]' "$logfile" | tail -3
}

# 5 contexts × 4 modes = 20 cells.  Order by ctx ascending so failures don't waste long-ctx wall time.
for ctx in 65536 131072 262144 524288 1048576; do
  for mode in none mtp_g2 mtp_g4 dflash_dm4; do
    run_cell "$mode" "$ctx" "${mode}_ctx${ctx}"
  done
done

# Compute energy + write results.csv (reusing the run_scientific.sh integrator).
python3 - <<'PY'
import os, re, csv, sys
LOGDIR = ".sisyphus/notes/gemma4-baseline/mtp-gamma/moe-scientific"
POWDIR = f"{LOGDIR}/power"
tags = {}
with open(f"{LOGDIR}/timestamps.csv") as f:
    for line in f:
        m = re.match(r"(\S+) rc=(\d+) t0=(\S+) t_end=(\S+)", line)
        if m:
            tags[m.group(1)] = {"rc": int(m.group(2)), "t0": float(m.group(3)), "t_end": float(m.group(4))}

with open(f"{LOGDIR}/results.csv", "w") as out:
    w = csv.writer(out)
    w.writerow([
        "cell","rc",
        "wall_s","prefill_ms","decode_ms","first_tok_ms",
        "prefill_tok_s","decode_tok_s",
        "AL","accept_rate","VRAM_GB",
        "avg_power_W","total_energy_J",
        "prefill_energy_J","decode_energy_J",
        "decode_J_per_tok",
    ])
    for tag, ts in tags.items():
        lp = f"{LOGDIR}/{tag}.log"; pp = f"{POWDIR}/{tag}.csv"
        if not os.path.exists(lp): continue
        log = open(lp).read()
        g = lambda pat: (re.search(pat, log).group(1) if re.search(pat, log) else "")
        prefill_ms = g(r"\[prefill\] \d+ tokens in ([0-9.]+) ms")
        prefill_tok_s = g(r"\[prefill\] \d+ tokens in [0-9.]+ ms \(([0-9.]+) tok/s\)")
        decode_ms = g(r"decode_ms=([0-9.]+)")
        decode_tok_s = g(r"tok/s=([0-9.]+)")
        first_tok_ms = g(r"first_tok_ms=([0-9.]+)")
        AL = g(r"mean_accept=([0-9.]+)") or g(r"avg_accept=([0-9.]+)")
        accept_rate = g(r"accept_rate=([0-9.]+)")
        VRAM = g(r"VRAM used=([0-9.]+) GB")
        samples = []
        if os.path.exists(pp):
            for L in open(pp):
                try:
                    t,p = L.strip().split(",")
                    samples.append((float(t), float(p)))
                except: pass
        wall_s = ts["t_end"] - ts["t0"]
        avg_power = (sum(p for _,p in samples)/len(samples)) if samples else 0.0
        total_E = 0.0
        for i in range(len(samples)-1):
            dt = samples[i+1][0]-samples[i][0]
            total_E += dt*(samples[i][1]+samples[i+1][1])/2.0
        pms = float(prefill_ms) if prefill_ms else 0.0
        dms = float(decode_ms) if decode_ms else 0.0
        if pms+dms > 0 and total_E > 0:
            # skip first 1s of model-load
            ai = max(0, int(len(samples)*1.0/max(wall_s,1)))
            active = samples[ai:]
            aE = 0.0
            for i in range(len(active)-1):
                dt = active[i+1][0]-active[i][0]
                aE += dt*(active[i][1]+active[i+1][1])/2.0
            prefill_E = aE * (pms/(pms+dms))
            decode_E  = aE * (dms/(pms+dms))
        else:
            prefill_E = decode_E = 0.0
        dJpt = decode_E/64.0 if decode_E>0 else 0.0
        w.writerow([
            tag, ts["rc"],
            f"{wall_s:.2f}", prefill_ms, decode_ms, first_tok_ms,
            prefill_tok_s, decode_tok_s,
            AL, accept_rate, VRAM,
            f"{avg_power:.1f}", f"{total_E:.1f}",
            f"{prefill_E:.1f}", f"{decode_E:.1f}",
            f"{dJpt:.3f}",
        ])
PY

echo "" >> $LOGDIR/SUMMARY.md
echo "## Results table" >> $LOGDIR/SUMMARY.md
echo '```' >> $LOGDIR/SUMMARY.md
column -t -s, $LOGDIR/results.csv >> $LOGDIR/SUMMARY.md 2>/dev/null
echo '```' >> $LOGDIR/SUMMARY.md
echo "DONE $(date +%H:%M:%S)" | tee -a $LOGDIR/SUMMARY.md
