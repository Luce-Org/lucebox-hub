# Tier 1 Drafter Speedup Proof — 2026-05-21

## Setup

- Target: Qwen3.6-27B-Q4_K_M, GPU: RTX 3090 24 GB
- Baseline drafter: Qwen3-0.6B-BF16 (1.5 GB, 28 layers, all 28 scored)
- Q8 drafter: Qwen3-0.6B-Q8_0 (0.8 GB, 28 layers, all 28 scored)
- Q8+L7: Qwen3-0.6B-Q8_0, only last 7 of 28 layers in tail-score reduction
- pflash keep_ratio=0.05, NIAH single-needle at 50% depth
- n_reps=3 per cell

## Results

| Condition | ctx  | drafter_fwd_p50 | ttft_p50 | NIAH | speedup_vs_baseline |
|-----------|------|-----------------|----------|------|---------------------|
| baseline  | 32K  | 11.42s          | 12.8s    | 100% | 1.0x                |
| baseline  | 64K  | 27.08s          | 29.4s    | 100% | 1.0x                |
| q8        | 32K  | 12.43s          | 14.0s    | 100% | 0.9x (slower)       |
| q8        | 64K  | 51.40s          | 54.3s    | 100% | 0.5x (slower)       |
| q8+L7     | 32K  | 22.46s          | 24.2s    | 100% | 0.5x (slower)       |
| q8+L7     | 64K  | 43.29s          | 46.8s    | 100% | 0.6x (slower)       |

Actual token counts: 32K prompt yields ~23K drafter tokens; 64K -> ~46K tokens.

## Timing Decomposition (server log, warm reps)

At 23K tokens:

| Step       | BF16 baseline | Q8+L7 (7 layers scored) |
|------------|---------------|-------------------------|
| A_compute  | 2.15s         | 5-7s (variable)         |
| FP kernel  | 1.95s         | 1.4-11s (variable)      |
| tail-score | 1.96s (28 L)  | 0.4-1.8s (7 L)          |
| total fwd  | ~11s          | ~7-38s                  |

Layer-subset scoring works: scoring time drops ~5x. But scoring is only 18% of
total. A_compute and FP dominate and Q8 does not accelerate them on this GPU.

## Root Cause

RTX 3090 BF16 tensor cores (312 TFLOPS) outperform Q8_0 path (142 TOPS INT8 +
dequant overhead). Q8_0 bypasses the fast `flash_prefill_forward_bf16` WMMA
kernel and falls back to ggml scalar path. High variance (7-38s same condition)
from 27B model competing for the same 24 GB VRAM.

## Verdict: Dark

Q8 drafter is slower than BF16 on RTX 3090 at these token counts. Layer-subset
scoring cuts tail-score 5x but that is only 18% of total forward cost.
NIAH retrieval preserved at 100% across all conditions.

What would help: early-exit forward (skip FFN on layers 22-27), a separate GPU
for the drafter, or keeping BF16 and optimizing the FP kernel instead.
