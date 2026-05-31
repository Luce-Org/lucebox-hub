# Plan: Hybrid Routed+Cold Path for All MoE Layers

## Problem
When `cold_compute` is enabled and layers have cold experts, DeltaNet layers are forced off the
routed fast path into the slower split path. This adds overhead from:
- Extra sync at split path entry (line 365)
- Hot graph rebuild/dispatch instead of reusing the pre-cached `rffn` graph
- Less pipelined execution

## Approach
Extend both the **routed fast path** (DeltaNet) and the **split-path routed FFN sub-path**
(attention layers) to handle cold experts inline — running the cold fused kernel on CPU in
parallel with the GPU routed FFN dispatch.

**Key property:** When all experts are hot at runtime (n_cold = 0), the cold branch is simply
skipped. The path executes identically to today's routed fast path — zero overhead.

## Changes

### 1. Routed Fast Path (DeltaNet layers, lines 260-355)

**Remove** the cold_compute guard from line 263:
```cpp
// Before:
&& !(state.cold_compute && !hybrid.layers[(size_t)il].cold_expert_ids.empty())
// After: removed — hybrid path handles cold inline
```

**Modify** the routing remap loop (lines 296-308) to also record cold IDs/weights.

**After remap, before rffn dispatch** — read ffn_post to CPU if cold compute needed.

**After rffn dispatch (async)** — run cold compute on CPU (parallel with GPU rffn).

**Combine** — upload cold result instead of zeroing.

### 2. Split-Path Routed FFN Sub-Path (attention layers, lines 441-495)

Same pattern: remove cold_compute guard, add cold partition + D2H + cold compute + upload.

### 3. Telemetry

Add `hybrid_cold_compute_us` counter for the new inline cold compute timing.

## Timing Analysis

- Current split path per mixed layer: ~1630µs
- Proposed hybrid: prefn(300µs) + max(cold(850µs), rffn(300µs)) + combine(50µs) ≈ 1200µs
- Conservative estimate: 5-10% throughput improvement
- All-hot case: zero overhead (cold branch skipped)
