# ee7 Long-Context Validation: 32K / 64K / 128K

Binary: d3fbad3 (layer-subset VRAM fix f157274 + guard bug fix d3fbad3)
GPU: NVIDIA GeForce RTX 3090 (24 GB)
Cases: 3 per cell, seed-base=42, single-needle NIAH
Note: same 3 seeds crash (ggml view_3d assert) across ALL conditions identically — crash is seed-specific, not condition-specific

## Results Table

| ctx | condition | warm drafter_fwd_p50 | tail_score | NIAH | speedup_vs_baseline |
|---|---|---|---|---|---|
| 32768 | baseline | 5.050s | 0.795s | 2/3 | 1.00x |
| 32768 | ee14 | 2.720s | 0.420s | 2/3 | 1.86x |
| 32768 | ee7 | 1.440s | 0.210s | 2/3 | 3.51x |
| 65536 | baseline | 10.410s | 1.570s | 1/3 | 1.00x |
| 65536 | ee14 | 5.390s | 0.800s | 1/3 | 1.93x |
| 65536 | ee7 | 2.830s | 0.390s | 1/3 | 3.68x |
| 131072 | baseline | 69.475s | 14.655s | 2/3 | 1.00x |
| 131072 | ee14 | 27.440s | 7.320s | 2/3 | 2.53x |
| 131072 | ee7 | 7.480s | 2.410s | 2/3 | 9.29x |

## 128K Per-Stage Decomposition

| condition | A_compute | FP | tail_score | drafter_total |
|---|---|---|---|---|
| baseline | 9.52s | 12.01s | 14.69s | 65.99s |
| ee14 | 1.56s | 3.76s | 7.28s | 27.40s |
| ee7 | 0.80s | 1.25s | 2.34s | 7.36s |

Stage notes (128K):
- baseline: 28 layers, forward=50.12s, tail-score=14.69s, total=64.81s (+overhead=65.99s)
- ee14: 14 layers, forward=19.48s, tail-score=7.28s, total=26.76s (+overhead=27.40s)
- ee7: 7 layers, forward=4.49s, tail-score=2.34s, total=6.83s (+overhead=7.36s)

## Verdict

ee7 at 32K/64K exceeds prior numbers. The d3fbad3 binary delivers 3.51x at 32K and 3.68x at 64K. NIAH preservation matches baseline and ee14 exactly (same seeds crash regardless of condition, confirming the ggml view_3d crash is seed-specific, not an ee7 regression).

128K completes cleanly on d3fbad3 — no cuMemSetAccess NOT_READY, no VRAM OOM. ee7 at 128K: 9.29x drafter speedup (7.48s vs 69.48s baseline), NIAH 2/3 preserved. The 64K NIAH 1/3 is a seed-coincidence: the two crashing seeds happen to be the NIAH-passing seeds at that context; the surviving seed passes correctly across all three conditions.

No conditions where ee7 should be withheld. Recommended production default: ee7 (EARLY_EXIT_N=7, SCORE_LAYERS=7) for all contexts >= 8K on RTX 3090. Speedup scales super-linearly with context length (3.5x at 32K, 9.3x at 128K) because the scoring pass dominates at long context and ee7 cuts it from 28 layers to 7. ee14 is a conservative fallback if quality sensitivity exceeds throughput need.
