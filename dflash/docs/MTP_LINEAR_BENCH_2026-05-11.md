# Linear MTP integrated decode — measured numbers, 2026-05-11

Validation run for `feat(dflash): linear native MTP integrated decode CLI`
(PR #154, stacked on PR #153 native MTP runtime).

## Setup

- **GPU**: NVIDIA RTX 6000 Ada Generation, sm_89, 48 GiB
- **Build**: Release, `DFLASH27B_USER_CUDA_ARCHITECTURES=89`,
  ninja + MSVC 19.44 + CUDA 12.8
- **Target model**: `Qwen3.6-27B-MTP-Q4_K_M.gguf`
  (`am17an/Qwen3.6-27B-MTP-GGUF` lineage)
- **Mode**: `--mtp-integrated --mtp-baseline-check --mtp-draft-n-max=2 --dflash-mtp-policy=always`
- **Context**: `--max-ctx=512`, `FA_WINDOW=2048` (default), KV `q8_0/q8_0` (default)
- **Prompt**: synthetic 64-token sequence of distinct ids
- **Method**: AR baseline (`run_target_ar_prompt`) vs integrated MTP loop
  (`run_mtp_integrated_prompt` with `mtp_draft_n_max=2`); same GGUF, same
  config; outputs compared token-by-token (`compare_ok` / `compare_fail`).

## Results

| n_gen | Baseline AR (tok/s) | MTP chain-2 (tok/s) | speed_ratio | MTP acceptance |
|---:|---:|---:|---:|---:|
| 64  | 9.48  | 12.04 | **1.270x** | 81.2% |
| 128 | 14.48 | 15.33 | **1.059x** | 81.2% |
| 256 | 15.11 | 17.38 | **1.151x** | 82.4% |

All three runs report `[mtp-baseline] compare_ok tokens=N mismatches=0`,
i.e. the integrated MTP loop produces the byte-identical greedy output to
the AR baseline on this prompt.

## Honest caveats

- **This is vs the AR baseline on the same MTP GGUF**, not against
  DFlash-classic + PFlash on a plain Qwen3.6-27B Q4_K_M. The AR baseline
  is slower than DFlash-classic; a PR-publishable speedup claim against
  DFlash-classic needs the `target_verify` graph-bucket cache work
  ("Waves A-D" / P1 in `MTP_ACCELERATION_ROADMAP_FOR_NEXT_AI_2026-05-11.md`).
- **Single-prompt result.** Workload sensitivity matters for MTP
  acceptance; published prompt families (P3 of the roadmap, multi-prompt
  gate) are pending.
- **Linear path only.** No DDTree hybrid, no immediate-bonus, no batched
  target verify, no bucketed graphs. This is the parity-correct floor
  the next PR builds on.
- **`--dflash-mtp-policy=auto`** is wired (P2 of the roadmap) but defaults
  to `always` for now since the smoke shows MTP wins at every measured
  `n_gen` against this baseline. The auto threshold (`min_n=192` default)
  remains conservative for prompt families that haven't been measured.

## Reproduce

```bash
# Build
cmake -S dflash -B dflash/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DDFLASH27B_USER_CUDA_ARCHITECTURES=89
cmake --build dflash/build --target test_dflash

# Synthetic prompt
python -c "import struct; \
ids = list(range(1000, 1064)); \
open('prompt.bin','wb').write(b''.join(struct.pack('<i', i) for i in ids))"

# Gate
./dflash/build/test_dflash \
  /path/to/Qwen3.6-27B-MTP-Q4_K_M.gguf \
  --mtp-integrated --mtp-baseline-check --mtp-draft-n-max=2 \
  --dflash-mtp-policy=always \
  --max-ctx=512 --n-gen=256 --prompt-file=prompt.bin --target-gpu=1
```

Or via the Python harness:

```bash
python dflash/scripts/mtp_baseline_gate.py \
  --target /path/to/Qwen3.6-27B-MTP-Q4_K_M.gguf \
  --prompt-file prompt.bin --n-gen 256 \
  --mtp-draft-n-max 2 --max-ctx 512 --gpu 1 \
  --min-speed-ratio 1.05 --out artifacts/mtp_n256.json
```
