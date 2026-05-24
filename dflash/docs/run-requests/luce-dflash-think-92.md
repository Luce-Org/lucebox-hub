# Run request: luce-dflash --think full-92 sweep at think_max ∈ {4k, 65k}

**Date opened**: 2026-05-24
**Status**: Needed to fill the 4th cell of the luce-dflash × OpenRouter think/no-think comparison matrix.

## Why

Today we have 3 of 4 cells:

| | --think | --no-think |
|---|---|---|
| **luce-dflash** | (missing) | RTX 5090 MaxQ 51/92 = 55.4% ✓ |
| **OpenRouter qwen3.6-27b** | 2026-05-21 58/92 = 63% ✓ | in flight (bragi running) |

A clean think/no-think A/B for luce-dflash requires a full 92-case --think
run. Prior evidence (RTX 5090 MaxQ 8-case L2-budget-sweep + RTX 3090 Ti
5/20-case abort) suggests --think hurts Qwen3.6 on this bench. Want to
confirm empirically at full N=92 to make the matrix complete.

## Asks

Two runs on whichever GPU is available (RTX 5090 MaxQ or sindri):

### Run 1 — constrained thinking (4k)

```bash
dflash_server \
  Qwen3.6-27B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 1236 \
  --max-ctx 131072 --prefix-cache-slots 0 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --draft dflash-draft-3.6-q8_0.gguf \
  --ddtree --ddtree-budget 22 \
  --think-max-tokens 4096
```

```bash
python3 dflash/scripts/bench_http_capability.py \
  --url http://127.0.0.1:1236 \
  --area ds4-eval \
  --max-tokens 5120 \
  --timeout 3600 \
  --think \
  --json-out dflash/docs/tuning-snapshots/<host>-qwen36-2026-05-24-think4k-full92/result.json \
  --trace .../trace.txt
```

`--max-tokens 5120` = `think_max(4096) + hard_limit_reply(512) + reply_pad(512)`.
Sized to allow phase-1 to fill 4k of reasoning then have ~1k for visible answer.

### Run 2 — max thinking (65k)

```bash
dflash_server ... --think-max-tokens 65536    # same other flags
```

```bash
python3 dflash/scripts/bench_http_capability.py \
  --url http://127.0.0.1:1236 \
  --area ds4-eval \
  --max-tokens 66560 \
  --timeout 3600 \
  --think \
  --json-out dflash/docs/tuning-snapshots/<host>-qwen36-2026-05-24-think65k-full92/result.json
```

## Output

Snapshot to `dflash/docs/tuning-snapshots/<host>-qwen36-2026-05-24-think{4k,65k}-full92/`
on integration. Update `SUMMARY-2026-05-24.md` with the new rows.

## Sindri side

An orchestrator at `/tmp/luce-dflash-think-92/run.sh` is armed on RTX 3090 Ti
and will run both phases sequentially after the current ds4-eval and AIME
sweeps complete (estimated ~Monday afternoon EDT). If RTX 5090 MaxQ can do
them faster, that side wins — sindri will skip phases the snapshot already
contains.
