# Run request: sindri RTX 3090 Ti Qwen3.6 `--no-think` full-92

**Date opened**: 2026-05-24
**Status**: Fills the missing RTX 3090 Ti cell of the qwen3.6-27b `--no-think`
×{OpenRouter, RTX 5090 MaxQ, RTX 3090 Ti} comparison.

## Why

Today we have two of three GPUs / hosting backends for qwen3.6-27b `--no-think`:

| Backend                       | Acc           | Avg wall/case | Avg comp toks | Tok/s overall |
|-------------------------------|---------------|---------------|---------------|---------------|
| Bragi RTX 5090 MaxQ 100W (Q4_K_M) | 51/92 = 55.4% | 105.6s        | 5108          | 48.4          |
| OpenRouter qwen3.6-27b (paid, full prec) | 53/92 = 57.6% | 173.8s        | 11139         | 64.1          |
| **Sindri RTX 3090 Ti 225W (Q4_K_M)** | **missing** | — | — | — |

Adding RTX 3090 Ti lets us:
1. Quantify the bragi-vs-sindri throughput delta at matched quant (Q4_K_M)
   without the OR full-precision confound.
2. Confirm whether `--no-think` accuracy is hardware-invariant
   (greedy sampling should give bit-identical answers ⇒ 51/92 on both
   bragi + sindri unless the quant rounds differently on Ampere vs
   Blackwell).
3. Compare prefill/decode tok/s now that sindri's `3b80fa8` lands the
   `usage.timings.{prefill_ms,decode_ms,decode_tokens_per_sec}` field —
   that didn't exist when the bragi 51/92 run was captured, so this is
   also the first chance to get the breakdown for either local GPU.

## Asks

Single run on sindri (RTX 3090 Ti 225W):

```bash
# Server side — same canonical Qwen3.6 config we use elsewhere
dflash_server \
  Qwen3.6-27B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 1236 \
  --max-ctx 131072 --prefix-cache-slots 0 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --draft dflash-draft-3.6-q8_0.gguf \
  --ddtree --ddtree-budget 22

# Bench side — match exactly what bragi ran for the 51/92 baseline:
python3 dflash/scripts/bench_http_capability.py \
  --url http://127.0.0.1:1236 \
  --area ds4-eval \
  --model dflash \
  --timeout 1800 \
  --no-think \
  --json-out dflash/docs/tuning-snapshots/sindri-rtx3090ti-qwen36-2026-05-24-nothink-full92/result.json \
  --trace dflash/docs/tuning-snapshots/sindri-rtx3090ti-qwen36-2026-05-24-nothink-full92/trace.txt
```

Notes:
- **Binary must include `3b80fa8`** (timings emit) so the per-row
  `timings.{prefill_ms,decode_ms,decode_tokens_per_sec}` lands and we get
  the prefill/decode split we're missing on bragi.
- **Binary should also include `56a4355`** (bench-side reasoning-tokens
  fallback) — not strictly required for `--no-think` since
  `reasoning_tokens` will be 0/None, but keeps the row schema consistent
  with the OR comparison data.
- Greedy sampler (bench sends `temperature=0`) means this is reproducible:
  bragi's 51/92 was greedy too. Any accuracy delta is purely hardware /
  quant rounding.

## Output

Snapshot to
`dflash/docs/tuning-snapshots/sindri-rtx3090ti-qwen36-2026-05-24-nothink-full92/`
on integration. Update `SUMMARY-2026-05-24.md` to fill the missing row.

## Bragi side

Bragi's image is currently `e9ced9e7` (built 2026-05-23 23:47, pre-`3b80fa8`).
The bragi sweep that produced 51/92 ran on that image, so its rows have
`timings: None`. A bragi rebuild + re-run is on deck once the current
`bragi_master.sh` sweep finishes; that snapshot will land at
`dflash/docs/tuning-snapshots/bragi-rtx5090laptop-qwen36-2026-05-25-nothink-full92-with-timings/`.

## 2026-05-25 addendum: KV cache config divergence

The `--cache-type-k q4_0 --cache-type-v q4_0` flags in the command above
were inherited from a 2-case A/B at
`tuning-snapshots/sindri-rtx3090ti-qwen36-2026-05-23-postmerge-{A,B}*`
(q4_0/q4_0 vs q8_0/q4_0). They were claimed to "match bragi" but
bragi's run-requests never set these flags, and the binary's auto-default
when `max_ctx > 6144` (on CUDA) is **`tq3_0`**, not `q4_0`. So this
"canonical config" has been a sindri-only override of the binary default
since 2026-05-23.

Of 98 historical snapshots, only 1 has a populated `server_info` block
(the post-`c35a8a4` bench script). So we have **no evidence** that bragi
was ever on q4_0. New runs from 2026-05-25 forward should drop the
explicit flags and let the binary auto-select tq3_0 — pending the A/B
designed at `dflash/docs/experiments/kv-cache-q4-vs-tq3-2026-05-25.md`.

The 53/92 = 57.6% snapshot remains valid as a measurement of "sindri at
q4_0/q4_0" — just don't read into it as evidence about default config.
