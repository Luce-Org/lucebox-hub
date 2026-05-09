#!/usr/bin/env bash
# Scientific bench: dense vs MoE × {code, creative} × dm ∈ {1,2,4,8,16,32}.
# For each cell: GPU power profile → total/prefill/decode energy + tok/J + tok/s.
# NOTE: no set -e; we want the script to keep going even if individual cells fail.
cd /home/peppi/Dev/lucebox-hub
export PATH=/usr/local/cuda-13.1/bin:$PATH

LOGDIR=.sisyphus/notes/gemma4-baseline/scientific
POWLOG=$LOGDIR/power
mkdir -p $LOGDIR $POWLOG

DENSE=models/gemma-4-31B-it-Q4_K_M.gguf
DENSE_DFLASH=dflash/models/draft-gemma4-31b/draft-q8_0.gguf
MOE=/home/peppi/models/gemma4-26b-a4b-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
MOE_DFLASH=/home/peppi/models/gemma4-26b-a4b-dflash/draft-q8_0.gguf

PROMPT_CODE=.sisyphus/notes/gemma4-baseline/prompts/humaneval_2.txt
PROMPT_CREATIVE=.sisyphus/notes/gemma4-baseline/prompts/long_open.txt

echo "# Scientific bench — $(date -Iseconds)" > $LOGDIR/SUMMARY.md
echo "Q8/Q8 KV, 4K ctx, n_predict=256, temp=0 seed=0 --ignore-eos, pflash on" >> $LOGDIR/SUMMARY.md

run_cell() {
  local model=$1; local draft=$2; local prompt=$3; local dm=$4; local tag=$5
  local logfile=$LOGDIR/${tag}.log
  local powfile=$POWLOG/${tag}.csv

  # Start GPU power telemetry: timestamp + power.draw, every 100ms via tight loop.
  ( while true; do
      printf "%s,%s\n" "$(date +%s.%N)" "$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)"
      sleep 0.1
    done ) > $powfile 2>/dev/null &
  local POW_PID=$!

  local t0=$(date +%s.%N)
  ./dflash/build/test_gemma4_dflash \
    --model $model \
    --draft $draft \
    --draft-method dflash --draft-max $dm \
    --tokens-file $prompt \
    --kv-k q8_0 --kv-v q8_0 \
    --ctx-size 4096 --pflash \
    --n-predict 256 --temp 0 --seed 0 --ignore-eos \
    > $logfile 2>&1 || true
  local rc=$?
  local t_end=$(date +%s.%N)

  kill $POW_PID 2>/dev/null || true
  wait $POW_PID 2>/dev/null || true

  echo "$tag rc=$rc t0=$t0 t_end=$t_end" >> $LOGDIR/timestamps.csv
}

# 24 cells: 2 models × 2 prompts × 6 dms
for dm in 1 2 4 8 16 32; do
  run_cell $DENSE $DENSE_DFLASH $PROMPT_CODE     $dm dense_code_dm${dm}
  run_cell $DENSE $DENSE_DFLASH $PROMPT_CREATIVE $dm dense_creative_dm${dm}
  run_cell $MOE   $MOE_DFLASH   $PROMPT_CODE     $dm moe_code_dm${dm}
  run_cell $MOE   $MOE_DFLASH   $PROMPT_CREATIVE $dm moe_creative_dm${dm}
done

# Analysis: parse logs + power profiles, compute per-phase energy
python3 - <<'PY' > $LOGDIR/results.csv
import os, re, csv
from glob import glob

LOGDIR = ".sisyphus/notes/gemma4-baseline/scientific"
POWDIR = f"{LOGDIR}/power"

# Parse timestamps.csv -> {tag: (t0, t_end)}
tags = {}
with open(f"{LOGDIR}/timestamps.csv") as f:
    for line in f:
        m = re.match(r"(\S+) rc=(\d+) t0=(\S+) t_end=(\S+)", line)
        if m:
            tags[m.group(1)] = {"rc": int(m.group(2)), "t0": float(m.group(3)), "t_end": float(m.group(4))}

writer = csv.writer(__import__("sys").stdout)
writer.writerow([
    "cell", "rc",
    "wall_s", "prefill_ms", "decode_ms", "first_tok_ms",
    "prefill_tok_s", "decode_tok_s",
    "AL", "VRAM_GB",
    "avg_power_W", "total_energy_J",
    "prefill_energy_J", "decode_energy_J",
    "decode_J_per_tok",
])

for tag, ts in tags.items():
    log_path = f"{LOGDIR}/{tag}.log"
    pow_path = f"{POWDIR}/{tag}.csv"
    if not os.path.exists(log_path):
        continue
    log = open(log_path).read()

    def grep(pat, default=""):
        m = re.search(pat, log)
        return m.group(1) if m else default

    prefill_ms = grep(r"\[prefill\] \d+ tokens in ([0-9.]+) ms", "")
    prefill_tok_s = grep(r"\[prefill\] \d+ tokens in [0-9.]+ ms \(([0-9.]+) tok/s\)", "")
    prefill_n = grep(r"prefill=(\d+) tokens", "")
    decode_ms = grep(r"decode_ms=([0-9.]+)", "")
    decode_tok_s = grep(r"tok/s=([0-9.]+)", "")
    first_tok_ms = grep(r"first_tok_ms=([0-9.]+)", "")
    AL = grep(r"avg_accept=([0-9.]+)", "")
    VRAM = grep(r"VRAM used=([0-9.]+) GB", "")

    # Power integration
    samples = []
    if os.path.exists(pow_path):
        for line in open(pow_path):
            try:
                t, p = line.strip().split(",")
                samples.append((float(t), float(p)))
            except:
                pass

    wall_s = ts["t_end"] - ts["t0"]
    avg_power = (sum(p for _, p in samples) / len(samples)) if samples else 0.0
    total_E = 0.0
    for i in range(len(samples) - 1):
        dt = samples[i+1][0] - samples[i][0]
        total_E += dt * (samples[i][1] + samples[i+1][1]) / 2  # trapezoidal

    # Per-phase energy: integrate over the binary's reported prefill/decode windows.
    # Phase boundaries from binary: T0 (start) → T0+startup_ms → T0+startup+prefill_ms → T_end.
    # We don't have explicit startup; approximate: first 1s is startup+model load (largely CPU, lower GPU power).
    # Simpler: split total energy by time fractions.
    pms = float(prefill_ms) if prefill_ms else 0.0
    dms = float(decode_ms) if decode_ms else 0.0
    total_active_ms = pms + dms
    if total_active_ms > 0 and total_E > 0:
        # Integrate over the *active* window (skip first ~1s of model load)
        active_start_idx = max(0, int(len(samples) * 1.0 / max(wall_s, 1)))
        active_samples = samples[active_start_idx:]
        active_E = 0.0
        for i in range(len(active_samples) - 1):
            dt = active_samples[i+1][0] - active_samples[i][0]
            active_E += dt * (active_samples[i][1] + active_samples[i+1][1]) / 2
        prefill_E = active_E * (pms / total_active_ms)
        decode_E = active_E * (dms / total_active_ms)
    else:
        prefill_E = decode_E = 0.0

    decode_J_per_tok = decode_E / 256.0 if decode_E > 0 else 0.0

    writer.writerow([
        tag, ts["rc"],
        f"{wall_s:.2f}", prefill_ms, decode_ms, first_tok_ms,
        prefill_tok_s, decode_tok_s,
        AL, VRAM,
        f"{avg_power:.1f}", f"{total_E:.1f}",
        f"{prefill_E:.1f}", f"{decode_E:.1f}",
        f"{decode_J_per_tok:.3f}",
    ])
PY

echo "" >> $LOGDIR/SUMMARY.md
echo "## Results table — see results.csv for full data" >> $LOGDIR/SUMMARY.md
echo '```' >> $LOGDIR/SUMMARY.md
column -t -s, $LOGDIR/results.csv >> $LOGDIR/SUMMARY.md 2>/dev/null
echo '```' >> $LOGDIR/SUMMARY.md
echo "DONE" | tee -a $LOGDIR/SUMMARY.md
