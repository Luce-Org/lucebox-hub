#!/usr/bin/env bash
# Matrix v3: validate the SWA-mask fix at HEAD across the production-relevant configs.
# Cells:
#   N1: target-only K=Q8 V=TQ3   (validate fix on long prompt; expect coherent)
#   N2: target-only K=Q8 V=Q8     (control, expect coherent — was M2 in v2)
#   N3: MTP        K=Q8 V=TQ3   (the production ship target — measure accept_rate)
#   N4: MTP        K=Q8 V=Q8     (previous safe baseline — was M4 in v2; expect crash ~step 210)
set -e
cd /home/peppi/Dev/lucebox-hub
export PATH=/usr/local/cuda-13.1/bin:$PATH
LOGDIR=.sisyphus/notes/gemma4-baseline/matrix-v3
mkdir -p $LOGDIR

MODEL=models/gemma-4-31B-it-Q4_K_M.gguf
MTP=models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q8_0.gguf
PROMPT=.sisyphus/notes/gemma4-baseline/prompts/long_open.txt

run_cell() {
  local tag=$1; local kvk=$2; local kvv=$3; local draft=$4
  local logfile=$LOGDIR/${tag}.log
  echo "=== $tag (K=$kvk V=$kvv draft=$draft) ===" | tee -a $LOGDIR/SUMMARY.md
  local args=(
    --model $MODEL
    --draft-method $draft
    --kv-k $kvk --kv-v $kvv
    --tokens-file $PROMPT
    --n-predict 256 --temp 0 --seed 0 --ignore-eos
  )
  if [ "$draft" = "mtp" ]; then
    args+=(--mtp $MTP)
  fi
  ./dflash/build/test_gemma4_dflash "${args[@]}" > $logfile 2>&1
  local rc=$?
  echo "$tag rc=$rc" | tee -a $LOGDIR/SUMMARY.md
}

echo "# Matrix v3 with SWA mask fix — $(date -Iseconds)" > $LOGDIR/SUMMARY.md
run_cell N1_none_q8_tq3 q8_0 tq3_0 none
run_cell N2_none_q8_q8  q8_0 q8_0  none
run_cell N3_mtp_q8_tq3  q8_0 tq3_0 mtp
run_cell N4_mtp_q8_q8   q8_0 q8_0  mtp

# Decode + report
echo "" | tee -a $LOGDIR/SUMMARY.md
echo "## Decoded outputs (first 80 generated tokens) + accept_rate trajectories" | tee -a $LOGDIR/SUMMARY.md
python3 - <<'PY' | tee -a $LOGDIR/SUMMARY.md
import re, os
from transformers import AutoTokenizer
LOGDIR = ".sisyphus/notes/gemma4-baseline/matrix-v3"
t = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

def extract_gen_ids(log):
    """Get generated token IDs from a log; tokens are space-separated ints between [prefill] and [stats]."""
    ids = []
    for line in log.splitlines():
        s = line.strip()
        if not s: continue
        if s.startswith("[") and s.endswith("]"): continue
        # Pure numeric line
        if re.match(r'^[\d\s]+$', s):
            for x in s.split():
                if x.isdigit() and int(x) < 262144:
                    ids.append(int(x))
    return ids

def extract_accept_rates(log):
    return [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r'\[mtp-step (\d+)\] accept_rate=([0-9.]+)', log)]

def extract_final_stats(log):
    m = re.search(r'\[mtp\] steps=(\d+) accepted=(\d+) accept_rate=([0-9.]+)', log)
    if m: return f"steps={m.group(1)} accepted={m.group(2)} accept_rate={m.group(3)}"
    m = re.search(r'\[stats\] generated=(\d+).*?tok/s=([0-9.]+)', log)
    if m: return f"generated={m.group(1)} tok/s={m.group(2)}"
    return "no stats"

for tag in ["N1_none_q8_tq3", "N2_none_q8_q8", "N3_mtp_q8_tq3", "N4_mtp_q8_q8"]:
    p = f"{LOGDIR}/{tag}.log"
    if not os.path.exists(p):
        print(f"\n### {tag}: NO LOG"); continue
    with open(p) as f: log = f.read()
    crashed = "GGML_ABORT" in log or "Aborted" in log or "core dumped" in log
    ids = extract_gen_ids(log)
    txt80 = t.decode(ids[:80], skip_special_tokens=False) if ids else "(no tokens)"
    print(f"\n### {tag}")
    print(f"crashed: {crashed}")
    print(f"final_stats: {extract_final_stats(log)}")
    print(f"first_80_decoded: {txt80!r}")
    if "mtp" in tag:
        rates = extract_accept_rates(log)
        if rates:
            print(f"accept_rate trajectory ({len(rates)} samples):")
            for step, rate in rates[::4]:  # every 4th sample
                print(f"  step={step:3d} rate={rate:.2f}")
            print(f"  step={rates[-1][0]:3d} rate={rates[-1][1]:.2f} (final)")
PY

echo "DONE" | tee -a $LOGDIR/SUMMARY.md
