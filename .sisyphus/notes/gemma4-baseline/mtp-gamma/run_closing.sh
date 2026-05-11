#!/usr/bin/env bash
# Close the matrix: Q8 ceiling cells (MoE long, Dense short) + drafter rows on top.
# All cells: pflash, --temp 0 --seed 0 --ignore-eos --n-predict 64.
# Per-cell GPU power telemetry @ 10 Hz → total/prefill/decode J + decode J/tok.

cd /home/peppi/Dev/lucebox-hub
export PATH=/usr/local/cuda-13.1/bin:$PATH

LOGDIR=.sisyphus/notes/gemma4-baseline/mtp-gamma/closing
POWDIR=$LOGDIR/power
mkdir -p $LOGDIR $POWDIR
: > $LOGDIR/timestamps.csv

MOE=/home/peppi/models/gemma4-26b-a4b-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
MOE_DFLASH=/home/peppi/models/gemma4-26b-a4b-dflash/draft-q8_0.gguf
DENSE=/home/peppi/Dev/lucebox-hub/models/gemma-4-31B-it-Q4_K_M.gguf
DENSE_DFLASH=/home/peppi/models/gemma4-31b-dflash/draft-q8_0.gguf
PROMPT=.sisyphus/notes/gemma4-baseline/prompts/long_code_50k.txt
DENSE_32K_PROMPT=.sisyphus/notes/gemma4-baseline/prompts/code_12k.txt

run_cell () {
  local model=$1 dflash_model=$2 ctx=$3 kv=$4 drafter=$5 tag=$6 prompt=$7
  local log=$LOGDIR/$tag.log
  local pow=$POWDIR/$tag.csv
  echo "=== $tag ($(date +%H:%M:%S)) ==="
  ( while true; do
      printf "%s,%s\n" "$(date +%s.%N)" "$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)"
      sleep 0.1
    done ) > "$pow" 2>/dev/null &
  local POW_PID=$!
  local args=( --model "$model" --ctx-size "$ctx" --n-predict 64 --temp 0 --seed 0 --ignore-eos --pflash --tokens-file "$prompt" )
  case $kv in
    q8)  args+=( --kv-k q8_0  --kv-v q8_0  ) ;;
    tq3) args+=( --kv-k tq3_0 --kv-v tq3_0 ) ;;
  esac
  case $drafter in
    none) ;;
    dflash:dm*) dm=${drafter##*:dm}; args+=( --draft-method dflash --draft "$dflash_model" --draft-max "$dm" ) ;;
  esac
  local t0=$(date +%s.%N)
  timeout 900 ./dflash/build/test_gemma4_dflash "${args[@]}" > "$log" 2>&1
  local rc=$?
  local te=$(date +%s.%N)
  kill $POW_PID 2>/dev/null; wait $POW_PID 2>/dev/null
  echo "$tag rc=$rc t0=$t0 t_end=$te" >> $LOGDIR/timestamps.csv
  grep -E '^\[stats\]|\[mem\]|avg_accept|prefill.*tok/s' "$log" | tail -4
}

# Q8 ceiling — no drafter, with power
run_cell "$MOE"   "" 131072 q8  none  moe_128K_q8_pf_none      "$PROMPT"
run_cell "$MOE"   "" 262144 q8  none  moe_256K_q8_pf_none      "$PROMPT"
run_cell "$MOE"   "" 524288 q8  none  moe_512K_q8_pf_none      "$PROMPT"
# Q8 + DFlash (test Q8 stack at long ctx)
run_cell "$MOE"   "$MOE_DFLASH" 131072 q8  dflash:dm4 moe_128K_q8_pf_dfl4 "$PROMPT"
run_cell "$MOE"   "$MOE_DFLASH" 262144 q8  dflash:dm4 moe_256K_q8_pf_dfl4 "$PROMPT"
run_cell "$MOE"   "$MOE_DFLASH" 524288 q8  dflash:dm4 moe_512K_q8_pf_dfl4 "$PROMPT"
# Dense Q8 with proper dm=16
run_cell "$DENSE" "$DENSE_DFLASH"  32768 q8  none      dense_32K_q8_pf_none "$DENSE_32K_PROMPT"
run_cell "$DENSE" "$DENSE_DFLASH"  65536 q8  none      dense_64K_q8_pf_none "$PROMPT"
run_cell "$DENSE" "$DENSE_DFLASH"  32768 q8  dflash:dm16 dense_32K_q8_pf_dfl16 "$DENSE_32K_PROMPT"
run_cell "$DENSE" "$DENSE_DFLASH"  65536 q8  dflash:dm16 dense_64K_q8_pf_dfl16 "$PROMPT"
# Dense TQ3 with proper dm=16
run_cell "$DENSE" "$DENSE_DFLASH"  32768 tq3 dflash:dm16 dense_32K_tq3_pf_dfl16 "$DENSE_32K_PROMPT"
run_cell "$DENSE" "$DENSE_DFLASH"  65536 tq3 dflash:dm16 dense_64K_tq3_pf_dfl16 "$PROMPT"

# Compute energy summary
python3 - <<'PY'
import os, re, csv
LOGDIR = ".sisyphus/notes/gemma4-baseline/mtp-gamma/closing"
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
            try: t,p = L.strip().split(","); samples.append((float(t), float(p)))
            except: pass
    rc, t0, te = ts.get(tag, (1, 0, 0))
    wall = te - t0
    avg_p = sum(p for _,p in samples)/len(samples) if samples else 0.0
    total_E = sum((samples[i+1][0]-samples[i][0])*(samples[i][1]+samples[i+1][1])/2 for i in range(len(samples)-1))
    decode_ms = float(g(r"decode_ms=([0-9.]+)") or 0)
    prefill_ms = float(g(r"\[prefill\] \d+ tokens in ([0-9.]+) ms") or 0)
    prefill_tok_s = g(r"\[prefill\] \d+ tokens in [0-9.]+ ms \(([0-9.]+) tok/s\)")
    if decode_ms+prefill_ms > 0 and total_E > 0:
        ai = max(0, int(len(samples)*1.0/max(wall,1)))
        active = samples[ai:]
        aE = sum((active[i+1][0]-active[i][0])*(active[i][1]+active[i+1][1])/2 for i in range(len(active)-1))
        pE = aE * (prefill_ms/(decode_ms+prefill_ms))
        dE = aE * (decode_ms/(decode_ms+prefill_ms))
    else:
        pE = dE = 0
    return {
        "rc": rc, "wall_s": f"{wall:.1f}",
        "decode_tps": g(r"tok/s=([0-9.]+)"),
        "prefill_tps": prefill_tok_s,
        "first_ms": g(r"first_tok_ms=([0-9.]+)"),
        "vram": g(r"VRAM used=([0-9.]+)"),
        "avg_W": f"{avg_p:.1f}",
        "total_J": f"{total_E:.1f}",
        "prefill_J": f"{pE:.1f}",
        "decode_J": f"{dE:.1f}",
        "decode_J_per_tok": f"{dE/64:.2f}" if dE > 0 else "",
    }

cells = ["moe_128K_q8_pf_none","moe_256K_q8_pf_none","moe_512K_q8_pf_none",
         "moe_128K_q8_pf_dfl4","moe_256K_q8_pf_dfl4","moe_512K_q8_pf_dfl4",
         "dense_32K_q8_pf_none","dense_64K_q8_pf_none",
         "dense_32K_q8_pf_dfl16","dense_64K_q8_pf_dfl16",
         "dense_32K_tq3_pf_dfl16","dense_64K_tq3_pf_dfl16"]
data = {c: parse(c) for c in cells}
hdr = ["cell","prefill tps","decode tps","first ms","VRAM","avg W","total J","prefill J","decode J","J/tok"]
rows = [hdr]
for c in cells:
    d = data[c]
    if not d: rows.append([c]+["—"]*9); continue
    rows.append([c, d["prefill_tps"], d["decode_tps"], d["first_ms"], d["vram"], d["avg_W"], d["total_J"], d["prefill_J"], d["decode_J"], d["decode_J_per_tok"]])
cw = [max(len(str(r[i])) for r in rows) for i in range(len(hdr))]
print()
for r in rows:
    print("  ".join(str(r[i]).ljust(cw[i]) for i in range(len(r))))
with open(f"{LOGDIR}/results.csv","w") as f:
    w = csv.writer(f); w.writerow(hdr)
    for r in rows[1:]: w.writerow(r)
print(f"\nresults.csv written to {LOGDIR}/results.csv")
PY
echo "=== done $(date +%H:%M:%S) ==="
