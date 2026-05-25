# KV cache q4_0 vs tq3_0 A/B — 2026-05-25

## Question

Sindri's canonical run-request forces `--cache-type-k q4_0 --cache-type-v q4_0`.
The binary's auto-default at our `max_ctx` (131072) is `tq3_0` on CUDA.
The "match bragi" rationale in the run-request was unverified — only 1
of 98 historical snapshots has populated `server_info`, so bragi's actual
KV cache type was never recorded. Was forcing `q4_0` actually helping,
hurting, or neutral vs the binary default?

## Hypothesis

- `q4_0` and `tq3_0` are both 3-4 bpw quants of the KV cache; both
  dequantize to FP16 on Ampere tensor cores for attention math.
- Memory footprint: `tq3_0` (3.0625 bpw on K/V) is slightly smaller
  than `q4_0` (4.5 bpw including the per-block scale). Modest VRAM
  savings (~30% on KV at long context).
- Throughput: dequant cost differs slightly; `tq3_0` has a denser
  unpack but smaller bandwidth footprint. Net effect is workload-
  dependent. Plausibly within ±5%.
- Accuracy: `tq3_0` has more quantization noise per K/V cell, which
  can drift attention scores at long context. On a hard-reasoning
  bench like ds4-eval, this could meaningfully change pass rate.
- Determinism: at temp=0, both should produce reproducible outputs
  across reruns *within the same KV type*. Cross-KV-type outputs
  may differ due to numerical noise.

## Experimental design

**Two phases, same 8 ds4-eval cases, same binary, same prompt set.**

### Phase A: `tq3_0` (binary default)

Drop the explicit `--cache-type-k/v` flags so the binary picks
`tq3_0` for our max_ctx.

```bash
dflash_server Qwen3.6-27B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 1236 \
  --max-ctx 131072 --prefix-cache-slots 0 \
  --draft dflash-draft-3.6-q8_0.gguf \
  --ddtree --ddtree-budget 22
```

### Phase B: `q4_0` (explicit, matches sindri canonical)

```bash
dflash_server Qwen3.6-27B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 1236 \
  --max-ctx 131072 --prefix-cache-slots 0 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --draft dflash-draft-3.6-q8_0.gguf \
  --ddtree --ddtree-budget 22
```

### Bench (both phases)

```bash
python3 dflash/scripts/bench_http_capability.py \
  --url http://127.0.0.1:1236 \
  --area ds4-eval --questions 8 \
  --max-tokens 16000 --timeout 1800 --no-think \
  --json-out <snapshot>/result.json \
  --trace    <snapshot>/trace.txt
```

8 cases × 2 phases ≈ 25-30 min wall.

## Pre-flight

- **Binary must include the 2026-05-25 /props.runtime expansion**
  (the patch in this PR — exposes `kv_cache_k`, `kv_cache_v`, `chunk`,
  `target_device`, `draft_device` for snapshot capture). The bench
  script's new `server_info.props` wholesale capture pins both phases'
  full configs in the JSON output so the A/B is self-documenting.
- Sindri must be free (whole `--think 92 v2` chain + AIME sweep done).

## Metrics

Per-phase:
- `passed/total` — accuracy
- `mean wall_s/case`
- `decode_tokens_per_sec` (per-row, from `usage.timings`)
- `[spec-decode] accepted=X/Y (Z%)` — accept rate
- `[spec-decode] avg_commit` — average commits per draft step
- VRAM peak (from `nvidia-smi --query-gpu=memory.used` polled during run)

Cross-phase:
- Per-case accuracy delta (which cases flip pass/fail)
- Per-case wall delta (>5% is signal at N=8 trials)
- Output-content hash delta (do same prompts produce different
  answers at temp=0 across KV types?)

## Outputs

Snapshots at:
```
dflash/docs/tuning-snapshots/sindri-rtx3090ti-2026-05-25-kv-q4-vs-tq3/
  phase-a-tq3_0/  result.json  trace.txt  server.log
  phase-b-q4_0/   result.json  trace.txt  server.log
  REPORT.md       # cross-phase analysis written after both finish
```

`REPORT.md` should answer:
1. Does forcing q4_0 win, lose, or tie on **pass rate**?
2. Does forcing q4_0 win, lose, or tie on **wall time**?
3. How many cases produce **different answers** between the two
   KV types at temp=0?
4. What should the canonical config be? Update run-request templates
   accordingly.

## Decision rule

- **q4_0 ≥ tq3_0 by ≥1pp pass rate AND comparable wall**: keep q4_0
  in run-request templates, document the rationale citing this A/B.
- **tq3_0 ≥ q4_0 by ≥1pp pass rate OR ≥5% wall improvement**: drop
  explicit `--cache-type-k/v` flags from all run-requests; let binary
  default win.
- **Tie within noise**: drop the explicit flags anyway (less surface
  area, matches binary intent, matches presumed bragi config).

## Out of scope

- Other KV types (`q8_0`, `f16`, `iq4_nl`) — settle the q4_0 vs tq3_0
  question first.
- Long-context regimes (>50k tokens prompt) — would need a different
  prompt set; do as a follow-up.
- Cross-host validation (run same A/B on bragi) — would triangulate
  whether KV type effects are hardware-specific. Defer.

## When this runs

Estimated start: after sindri `--think 92 v2` phase 2 + AIME sweep
finishes (~2026-05-26 evening to 2026-05-27 morning EDT, given the
24-48h phase-2 ETA from the 09:19 EDT 2026-05-25 boot).
