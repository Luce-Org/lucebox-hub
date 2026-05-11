#!/usr/bin/env bash
# One-Variable-At-A-Time ablation at the reference cell.
# Reference: MoE 26B-A4B × ctx=65536 × long_code_50k.txt × n_predict=64 × --temp 0 --seed 0 --ignore-eos.
# Outputs: per-cell logs + power CSVs + summary table with deltas vs naive baseline.

cd /home/peppi/Dev/lucebox-hub
export PATH=/usr/local/cuda-13.1/bin:$PATH

LOGDIR=.sisyphus/notes/gemma4-baseline/mtp-gamma/ovat-moe-64k-code
POWDIR=$LOGDIR/power
mkdir -p $LOGDIR $POWDIR
: > $LOGDIR/timestamps.csv

MOE=/home/peppi/models/gemma4-26b-a4b-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
MOE_MTP=/home/peppi/models/gemma4-mtp-26b-a4b/gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf
MOE_DFLASH=/home/peppi/models/gemma4-26b-a4b-dflash/draft-q8_0.gguf
PROMPT=.sisyphus/notes/gemma4-baseline/prompts/long_code_50k.txt
CTX=65536

# Each cell: positional args (tag, kv, pflash, drafter_mode, drafter_arg)
# kv:        q8|tq3
# pflash:    on|off
# drafter:   none|dflash:dm4|dflash:dm8|dflash:dm16|mtp:g1|mtp:g2|mtp:g4
run_cell () {
  local tag=$1 kv=$2 pflash=$3 drafter=$4
  local log=$LOGDIR/${tag}.log
  local pow=$POWDIR/${tag}.csv
  echo "=== ${tag} ($(date +%H:%M:%S)) ==="
  ( while true; do
      printf "%s,%s\n" "$(date +%s.%N)" "$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)"
      sleep 0.1
    done ) > "$pow" 2>/dev/null &
  local POW_PID=$!

  local args=( --model "$MOE" --ctx-size "$CTX" --n-predict 64 --temp 0 --seed 0 --ignore-eos --tokens-file "$PROMPT" )
  case $kv in
    q8)  args+=( --kv-k q8_0  --kv-v q8_0  ) ;;
    tq3) args+=( --kv-k tq3_0 --kv-v tq3_0 ) ;;
  esac
  [[ $pflash == on ]] && args+=( --pflash )
  case $drafter in
    none) ;;
    dflash:dm*) dm=${drafter##*:dm}; args+=( --draft-method dflash --draft "$MOE_DFLASH" --draft-max "$dm" ) ;;
    mtp:g*)     gm=${drafter##*:g};  args+=( --draft-method mtp --mtp "$MOE_MTP" --gamma "$gm" ) ;;
  esac

  local t0=$(date +%s.%N)
  timeout 900 ./dflash/build/test_gemma4_dflash "${args[@]}" > "$log" 2>&1
  local rc=$?
  local te=$(date +%s.%N)
  kill $POW_PID 2>/dev/null; wait $POW_PID 2>/dev/null
  echo "$tag rc=$rc t0=$t0 t_end=$te" >> $LOGDIR/timestamps.csv
  grep -E '^\[stats\]|\[mem\]|accept_rate|avg_accept' "$log" | tail -3
}

# Cell 0: naive baseline Q8/Q8 + no pflash + no drafter
run_cell 00_naive_q8     q8  off none
# Cell 1: + TQ3 KV (Q8→TQ3, no pflash, no drafter)
run_cell 01_tq3          tq3 off none
# Cell 2: + pflash on top of Q8 (Q8, pflash, no drafter)
run_cell 02_q8_pflash    q8  on  none
# Cell 3: TQ3 + pflash, no drafter — the foundation for drafter cells
run_cell 03_tq3_pflash   tq3 on  none
# Cell 4-6: TQ3 + pflash + DFlash dm = 4/8/16
run_cell 04_tq3_pf_dfl4  tq3 on  dflash:dm4
run_cell 05_tq3_pf_dfl8  tq3 on  dflash:dm8
run_cell 06_tq3_pf_dfl16 tq3 on  dflash:dm16
# Cell 7-9: TQ3 + pflash + MTP γ = 1/2/4
run_cell 07_tq3_pf_mtp1  tq3 on  mtp:g1
run_cell 08_tq3_pf_mtp2  tq3 on  mtp:g2
run_cell 09_tq3_pf_mtp4  tq3 on  mtp:g4

# Compute summary with deltas vs baseline + decode J/tok
python3 - <<'PY'
import os, re, csv
LOGDIR = ".sisyphus/notes/gemma4-baseline/mtp-gamma/ovat-moe-64k-code"
POWDIR = f"{LOGDIR}/power"
ts = {}
with open(f"{LOGDIR}/timestamps.csv") as f:
    for line in f:
        m = re.match(r"(\S+) rc=(\d+) t0=(\S+) t_end=(\S+)", line)
        if m: ts[m.group(1)] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))

def parse(tag):
    p = f"{LOGDIR}/{tag}.log"
    if not os.path.exists(p): return None
    log = open(p).read()
    g = lambda pat: (re.search(pat, log).group(1) if re.search(pat, log) else "")
    samples = []
    pp = f"{POWDIR}/{tag}.csv"
    if os.path.exists(pp):
        for L in open(pp):
            try:
                t,p = L.strip().split(","); samples.append((float(t), float(p)))
            except: pass
    rc, t0, te = ts.get(tag, (1, 0, 0))
    wall = te - t0
    avg_p = sum(p for _,p in samples)/len(samples) if samples else 0.0
    total_E = sum((samples[i+1][0]-samples[i][0])*(samples[i][1]+samples[i+1][1])/2 for i in range(len(samples)-1))
    decode_ms = float(g(r"decode_ms=([0-9.]+)") or 0)
    prefill_ms = float(g(r"\[prefill\] \d+ tokens in ([0-9.]+) ms") or 0)
    if decode_ms+prefill_ms > 0 and total_E > 0:
        ai = max(0, int(len(samples)*1.0/max(wall,1)))
        aE = sum((samples[i+1][0]-samples[i][0])*(samples[i][1]+samples[i+1][1])/2 for i in range(ai, len(samples)-1))
        dE = aE * (decode_ms/(decode_ms+prefill_ms))
    else: dE = 0
    return {
        "rc": rc, "wall_s": f"{wall:.1f}",
        "tps": g(r"tok/s=([0-9.]+)"),
        "ftm": g(r"first_tok_ms=([0-9.]+)"),
        "acc": g(r"accept_rate=([0-9.]+)") or g(r"avg_accept=([0-9.]+)"),
        "vram": g(r"VRAM used=([0-9.]+)"),
        "prefill_tps": g(r"\[prefill\] \d+ tokens in [0-9.]+ ms \(([0-9.]+) tok/s\)"),
        "avg_W": f"{avg_p:.1f}",
        "dec_J_per_tok": f"{dE/64:.2f}" if dE > 0 else "",
    }

cells = ["00_naive_q8","01_tq3","02_q8_pflash","03_tq3_pflash",
         "04_tq3_pf_dfl4","05_tq3_pf_dfl8","06_tq3_pf_dfl16",
         "07_tq3_pf_mtp1","08_tq3_pf_mtp2","09_tq3_pf_mtp4"]
data = {c: parse(c) for c in cells}

# Pretty table
hdr = ["cell","prefill tps","decode tps","first ms","accept","VRAM","avg W","J/tok"]
rows = [hdr]
for c in cells:
    d = data[c]
    if not d:
        rows.append([c]+["—"]*7); continue
    rows.append([c, d["prefill_tps"], d["tps"], d["ftm"], d["acc"], d["vram"], d["avg_W"], d["dec_J_per_tok"]])

# Compute column widths
cw = [max(len(str(r[i])) for r in rows) for i in range(len(hdr))]
out = []
for r in rows:
    out.append("  ".join(str(r[i]).ljust(cw[i]) for i in range(len(r))))
print("\n" + "\n".join(out))

# Deltas vs naive
baseline_tps = float(data["00_naive_q8"]["tps"] or 0)
print(f"\n=== Decode tok/s delta vs naive (cell 00) ===")
for c in cells[1:]:
    d = data[c]
    if not d or not d["tps"]: continue
    delta = float(d["tps"]) - baseline_tps
    mult = float(d["tps"]) / baseline_tps if baseline_tps > 0 else 0
    print(f"  {c}: {float(d['tps']):.2f} ({delta:+.2f}, {mult:.2f}x)")

with open(f"{LOGDIR}/results.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    for r in rows[1:]:
        w.writerow(r)
print(f"\nresults.csv written")
PY
echo "=== done $(date +%H:%M:%S) ==="
