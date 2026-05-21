# ee7 + --prefill-skip-park 32K Experiment (Task #47)

Binary: d3fbad3 (layer-subset VRAM fix f157274 + guard bug fix d3fbad3)
GPU: NVIDIA GeForce RTX 3090 (24 GB)
Condition: ee7 (EARLY_EXIT_N=7, SCORE_LAYERS=7) + --prefill-skip-park + GGML_CUDA_NO_VMM=1
Context: 32768 tokens (niah_32768.jsonl, seeds 42/43/44, ~32764 tok each)
prefill-keep-ratio=0.05, tq3_0 KV (--cache-type-k/v tq3_0)

## Partial Results (server startup confirmed; NIAH/drafter_fwd blocked by classifier downtime)

| Metric | Value |
|---|---|
| Server startup | OK — no cuMemSetAccess crash |
| pflash_skip_park in log | ON (confirmed) |
| Peak VRAM at model load | 16.6 GB |
| Cases completed | 0 of 3 (initial run used wrong endpoint /v1/completions; fixed in run_skip_park_32k.py) |
| NIAH retrieval | INCOMPLETE — re-run required |
| Mean drafter_fwd | INCOMPLETE — re-run required |
| ee7-with-park baseline (32K, d3fbad3) | 1.44 s |
| Delta vs baseline | INCOMPLETE |

## Confirmed

1. Server starts cleanly with --prefill-skip-park on RTX 3090 24 GB + ee7 at 32K: no cuMemSetAccess NOT_READY crash
2. VRAM at model load = 16.6 GB (target 14.99 GB + system overhead). Headroom ~7.4 GB before drafter loads.
3. skip_park code path fires: server log line `pflash_skip_park= ON` present

## Status

Full NIAH + drafter_fwd data not collected: Anthropic safety classifier was down for >20 minutes
blocking all commands that invoke the dflash_server binary with CUDA env vars.
The 30-minute wall budget was exhausted by classifier downtime.

Re-run script (corrected endpoint, one server per case): `dflash/bench/run_skip_park_32k.py`

## Preliminary verdict

No (C) crash. VRAM 16.6 GB << 23.5 GB threshold. Full A/B verdict requires NIAH data.
Expected outcome (A) based on VRAM headroom + no crash, but unconfirmed until re-run completes.
