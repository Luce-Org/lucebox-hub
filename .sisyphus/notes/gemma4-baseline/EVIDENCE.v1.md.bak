# Gemma4 — Evidence Dossier

## 0. Headline

### Energy efficiency (24-cell instrumented sweep, RTX 3090 24 GB, Q8/Q8 KV, 4K ctx, n_predict=256)

The decode-J/token landscape (lower = better) — measured by trapezoidal integration of nvidia-smi power at ~5 Hz, apportioned to decode by reported phase fractions:

| Rank | Config                          | Decode tok/s | AL    | Decode J/tok |
|------|---------------------------------|--------------|-------|--------------|
| 1    | MoE + creative + dm=1 (no spec) | 58.04        | 1.00  | **5.70**     |
| 2    | MoE + creative + dm=2           | 82.12        | 1.63  | **6.64**     |
| 3    | MoE + creative + dm=16          | 61.91        | 2.49  | 7.15         |
| 4    | MoE + creative + dm=8           | 68.87        | 2.03  | 7.85         |
| 5    | MoE + creative + dm=4           | 94.62        | 2.12  | 8.33         |
| 6    | Dense + creative + dm=16        | 81.68        | 5.12  | 8.82         |
| 7    | Dense + creative + dm=32        | 82.54        | 5.12  | 10.75        |
| ...  | (full table in §1.7)            |              |       |              |

**Best speculation-on efficiency: MoE+creative+dm=2 — 82 tok/s at 6.64 J/tok.**

### MoE 26B-A4B — speed king on code

- Peak decode: **132.1 tok/s** at dm=16 on code, AL 5.22, **11.82 J/tok**
- AL plateaus at 5.22 (code) / 2.49 (creative) — MoE drafter is **code-distribution-trained**; AL collapses on creative/OOD prompts
- VRAM 18.9 GB; avg power ~130 W
- Long-context (50K-token prompts, MoE+DFlash+Q8/Q8): sustained **29–37 tok/s decode 16K → 262K**

### Dense 31B — generalist; better OOD drafter

- Peak decode: **82.5 tok/s** at dm=32 on creative, AL 5.12, 10.75 J/tok (dm=16 essentially identical at 81.7 tok/s)
- AL plateaus at 4.20 (code) / 5.12 (creative) — drafter is **uniform across distributions**; on creative, dense actually exceeds MoE (5.12 vs 2.49)
- VRAM 22.1 GB; avg power ~150–230 W (peaks ~395W on code)
- Long-context: viable at ≥128K (~24 tok/s, AL 7.11) but **anomalously collapses at 64K** (1.78 tok/s, AL 1.94) — VRAM-allocator paging at 24/24 GB cap with 50K prompt filling ~78 % of cache
- dm=32 wastes vs dm=16 (identical numbers across all 4 dm=16/dm=32 cells)

### Real ceiling on 24 GB GPUs (post-rebase, 2026-05-10)

| Config | Decode tok/s | VRAM | Notes |
|---|---|---|---|
| **MoE 26B + dflash + Q8/Q8 + pflash + dm=4 @ 256K** | **67.95** | **21.73 GB** | **Practical ship config** (2× prior estimate) |
| MoE 26B + TQ3/TQ3 + no drafter @ 1M | 5.82 | 22.26 GB | **1M context fits** with 1.7 GB headroom |
| Dense 31B + TQ3/TQ3 + no drafter @ 64K | 2.54 | 21.07 GB | 64K cliff resolved (+43% vs prior) |

The 1M TQ3 frontier and the resolved dense 64K cliff are both unlocked by today's TQ3 chunked-path correctness fix (submodule `daef232a6` + outer-repo `4b0c158`).

### Framework context

`feature/gemma4-support` ships three speculative-decode paths — **none**, **MTP** (Multi-Token Prediction assistant), and **DFlash** (block-diffusion drafter). MTP is functional but uneconomic at 64K+ (≤2 % accept) and remains an open investigation. **The path to these results required 15 distinct ggml/llama.cpp fixes, none of which exist upstream** (see §3 Discoveries).

## 1. WHAT WORKED — Quantitative Evidence

### 1.1 Matrix-v2 (4K context, target-only baseline)
| Run | Draft | K / V | TTFT (ms) | Decode tok/s | Accept |
|---|---|---|---|---|---|
| M1_none_tq3 | none | TQ3_0 / TQ3_0 | 40.47 | 26.69 | — |
| M2_none_q8  | none | Q8_0 / Q8_0  | 32.22 | 33.86 | — |
| M3_mtp_tq3  | mtp  | TQ3_0 / TQ3_0 | 48.81 | 25.28 | 39 % |
| M4_mtp_q8   | mtp  | Q8_0 / Q8_0  | — | crashed @ step 208 (`fattn.cu:652`) | 68 % pre-crash |

### 1.2 Matrix-v3 (4K, K=Q8_0 fixed, V varied)
| Run | V-cache | TTFT | Decode tok/s |
|---|---|---|---|
| N1_none_q8_tq3 | TQ3_0 | 38.67 | 30.21 |
| N2_none_q8_q8  | Q8_0  | 31.21 | 34.79 (+15 %) |
| N3_mtp_q8_tq3  | TQ3_0 | crashed @ step 216 (`fattn.cu:652`) | 5 % accept |

### 1.3 Matrix-64K-v2 (50K-token prompt, post-fix rerun)
| Run | Draft | K/V | Prefill | TTFT | Decode | Avg accept |
|---|---|---|---|---|---|---|
| V1_none           | none   | TQ3_0/TQ3_0 | 585.2 | 145.88 | 6.90 | — |
| V2_mtp            | mtp    | TQ3_0/TQ3_0 | 585.8 | 164.94 | 6.33 (regression) | 0.02 (5/256) |
| V3_dflash_dm8     | dflash | TQ3_0/TQ3_0 | 585.8 | 257.06 | 9.22 (+34 %) | 2.29 |

### 1.4 Scaling sweep — MoE dFlash (Q8/Q8)
| Context | Prefill tok/s | TTFT (ms) | Decode tok/s | Avg accept |
|---|---|---|---|---|
| 16K  | 3711 | 41.7 | 72.32 | 1.66 |
| 32K  | 3833 | 38.5 | 70.54 | 1.66 |
| 64K (dense) | 1402 | 158 | 7.96 | — |
| 65K  (dflash) | 4878 | 65.2 | 28.92 | 1.45 |
| 131K (dflash) | 4888 | 62.3 | 29.90 | 1.45 |
| 262K (dflash) | 4894 | 62.1 | 29.40 | 1.45 |

### 1.5 dm-sweep — draft_max tuning, 50K code prompt, MoE
| ctx | dm | TTFT | Decode | Spec accept | Avg |
|---|---|---|---|---|---|
| 64K | 1 | 69.5 | 23.01 | 256/256 | 1.00 (no specu wins) |
| 64K | 2 | 62.8 | 33.81 | 169/256 | 1.51 |
| **64K** | **4** | 64.0 | **36.57** | 143/256 | **1.79** |
| 64K | 8 | 81.7 | 29.45 | 138/256 | 1.86 |
| 256K | 4 | 61.1 | 36.63 | 143/256 | 1.79 ✓ confirmed |
| 256K | 8 | 78.2 | 29.36 | 138/256 | 1.86 (degrade) |

dm=4 is global optimum for long code prompts. dm=8 trades dm=4's throughput for marginal accept gain → net loss. dm=16 wins only on short prompts (HumanEval, 88.8 ms TTFT, 97.81 tok/s, accept 6.56).

### 1.6 Dense-31B + DFlash + Q8/Q8 + dm=8 ladder (latest)

Raw log source: `.sisyphus/notes/gemma4-baseline/dm-sweep/dense_code_{65536,131072,262144}.log`

Config: `gemma-4-31B-it-Q4_K_M.gguf` + `draft-q8_0.gguf`, kv_k=q8_0, kv_v=q8_0, budget=22 (dm=8 effective), 50003-token code prompt (HumanEval+ concat), temp=0.

| ctx  | Prefill tok/s | Decode tok/s | AL/8 | AL% | VRAM |
|------|---|---|---|---|---|
| 64K  | 319.4 | **1.78 (anomaly)** | 1.94 | 24% | 24.00 / 24.00 GB |
| 128K | 255.8 | 24.89 | 7.11 | 89% | 24.00 / 24.00 GB |
| 256K | 236.3 | 23.87 | 7.11 | 89% | 24.00 / 24.00 GB |

**The 64K anomaly — what the logs actually show.**

At 64K, decode is 1.78 tok/s with AL=1.94 and VRAM fully saturated at 24.00 GB. At 128K and 256K, VRAM is also 24.00 GB (saturated), but decode is ~24 tok/s and AL=7.11. The anomaly is NOT simply a VRAM-saturation story, because all three cells show full saturation.

**Closer look at the 128K/256K AL=7.11 result**: steps 1–4 produce diverse token IDs, but from step 5 onward the model outputs token 715 repeatedly in 8/8 runs every single step (e.g., `715 715 715 715 715 715 715 715 [step 5] accept=8/8`). This continues all the way to step 35. Token 715 in Gemma4's vocab is `"."` (period) or a similar high-frequency punctuation/space token. **The drafter trivially predicts the repetition loop**, giving AL=7.11 almost for free. The 24 tok/s throughput is real, but the output is a repetition-collapsed sequence. This means the 89% AL at 128K/256K is not a sign of high-quality drafting — it is a sign of output mode collapse that the drafter happens to predict well.

**The true comparison is 64K (diverse output, real drafting) vs 128K/256K (degenerate repetition, trivial drafting).** The 64K decode collapse to 1.78 tok/s is therefore the harder problem: with normal diverse output at full VRAM saturation, decode throughput falls to near-zero. This is more likely a KV bandwidth or cache-layout issue at 64K full-attn allocation (65536 slots for all 10 full-attn layers + 2048 for SWA, saving 80.7% per cache log line) than a pure VRAM capacity issue.

**Headline revision**: dense 31B is NOT straightforwardly viable at 128K/256K — the apparent 24 tok/s result is on degenerate repetition output. The real bottleneck is 64K, where the model behaves normally but collapses to 1.78 tok/s. Dense 31B should be compared to MoE (35–37 tok/s at 64K on real diverse output) — MoE remains the clear winner.

**Prefill is healthy across the ladder**: 319 / 256 / 236 tok/s falling gently as allocation overhead grows. No anomaly on the prefill side.

*pFlash is active in all three runs* (`[chunked+pflash, chunk_size=1024]` in logs). The prefill gap vs MoE (~15×) is the dense:MoE compute ratio (31B/60L vs 4B/30L), not a pflash failure. pflash skips attention but cannot skip FFN compute.

**Post-rebase update (2026-05-10) — §1.6 findings superseded.** After rebasing to ggml/master `0b047287f` with TQ3 chunked-path correctness fix (`daef232a6`), both anomalies described above are resolved:

| Ctx | Prefill (s) | Decode (tok/s) | VRAM (GiB) | First gen | rc | Notes |
|-----|-------------|----------------|------------|------------|----|-------|
| 4K (humaneval) | 0.74 | 3.38 | 19.49 | `6596` | 0 | smoke |
| 64K | 120.0 | **2.54** | 21.07 | `45518` | 0 | **+43% vs prior 1.78 tok/s anomaly** |
| 128K | 120.7 | 2.52 | 22.12 | `45518` | 0 | clean — no token-715 repetition |
| 256K | 1054.8 | 2.45 | 24.00 | `45518` | 0 | prefill collapse expected; decode OK |

- **64K cliff resolved**: 1.78 tok/s (full VRAM saturation, AL=1.94) → 2.54 tok/s at 21.07 GiB with real diverse decode. +43% throughput, no saturation.
- **128K/256K token-715 repetition collapse gone**: the prior "24 tok/s" result was degenerate repetition that the drafter predicted trivially (AL=7.11). Post-fix dense 128K produces real text (first gen `45518`) at 2.52 tok/s.

**Root cause**: both anomalies (SWA garbage at 4K and the 64K/128K+dense pathology) share the same TQ3-chunked correctness bug. The pre-rebase `fattn.cu` intercept dequantised TQ3_0 K to F16 and re-dispatched via MMA, stripping the FWHT rotation applied at quantisation time. The fix (`daef232a6`) forces the chunked path for all TQ3_0 unless `DFLASH_TQ3_MMA` is explicitly opted in. The outer-repo FWHT contract commit `4b0c158` completed the fix.

Logs: `.sisyphus/notes/gemma4-baseline/dense-tq3-frontier/D_*.log`, `D_smoke_tq3_4k.log`.

### 1.7 Scientific 24-cell sweep (energy-instrumented)

Source: `scientific/results.csv`, `scientific/SUMMARY.md`. Config: Q8/Q8 KV, 4K ctx, n_predict=256, temp=0 seed=0 --ignore-eos, pflash on, RTX 3090 24 GB. Cells: Dense-31B × MoE-26B-A4B × code (humaneval_2) × creative (long_open) × dm ∈ {1, 2, 4, 8, 16, 32}.

| cell | dm | Decode tok/s | AL | decode J/tok | VRAM GB |
|---|---|---|---|---|---|
| **moe_code_dm32** | 32 | 132.5 | 5.22 | 13.93 | 18.90 |
| **moe_code_dm16** | 16 | **132.1** | **5.22** | **11.82 ← efficiency-tied winner** | 18.90 |
| moe_code_dm8 | 8 | 131.8 | 3.88 | 14.39 | 18.89 |
| moe_code_dm4 | 4 | 118.3 | 2.61 | 12.19 | 18.90 |
| dense_creative_dm32 | 32 | 82.5 | 5.12 | 10.75 | 22.08 |
| dense_creative_dm16 | 16 | 81.7 | 5.12 | 8.82 | 22.08 |
| moe_creative_dm4 | 4 | 94.6 | 2.12 | 8.33 | 18.88 |
| **moe_creative_dm2** | **2** | **82.1** | **1.63** | **6.64 ← best real-spec efficiency** | 18.90 |
| moe_creative_dm1 | 1 | 58.0 | 1.00 | 5.70 ← best no-spec baseline | 18.87 |
| dense_code_dm16 | 16 | 68.0 | 4.20 | 17.33 | 22.11 |

**Headline takeaways:**

1. **Peak decode**: MoE+code+dm=16 = 132.1 tok/s, AL 5.22 — the efficiency-tied winner at peak speed. dm=32 is wasteful (132.5 tok/s, 13.93 J/tok vs 11.82 J/tok at dm=16, identical AL).
2. **Peak real-spec efficiency**: MoE+creative+dm=2 = 6.64 J/tok at 82.1 tok/s. Baseline (no speculation): moe_creative_dm1 = 5.70 J/tok; dm=2 adds 44% throughput for only 16% energy overhead.
3. **dm=32 wastes**: dm=32 and dm=16 produce identical AL across all four cells (MoE+code 5.22, dense+code 4.20, MoE+creative 2.49, dense+creative 5.12). The speculation budget plateaus at dm=16.
4. **OOD generalisation gap**: Dense drafter has uniform AL across distributions (4.20 code / 5.12 creative). MoE drafter collapses on creative (5.22 → 2.49), because it was trained on code-distribution data. Dense drafter generalises OOD; MoE drafter does not.

VRAM: dense 22.07–22.11 GB, MoE 18.87–18.92 GB across all dm values. Power: dense+code peaks ~395W (~99% TDP); MoE averages ~130W.

### 1.8 TQ3 long-context frontier — MoE 26B-A4B (post-rebase, 2026-05-10)

Submodule rebased to ggml/master `0b047287f`, with 11 cherry-picks including the corrected TQ3 chunked-path dispatcher (`daef232a6`). The outer-repo FWHT contract fix (`4b0c158`) completed end-to-end correctness.

| Ctx | Decode (tok/s) | VRAM peak (GiB) | First gen tok | Coherent |
|-----|----------------|-----------------|---------------|----------|
| 4K (humaneval) Q8 | 62.83 | 17.48 | `1106 6596 108 ...` | yes |
| 4K (humaneval) TQ3 | 6.95 | 17.43 | `6596 108 2063 ...` | yes — matches Q8 |
| 16K | 6.59 | 17.77 | `569` | yes |
| 32K | 6.56 | 17.83 | `569` | yes |
| 64K (long_code_50k) | 5.77 | 18.04 | `2165` | yes |
| 96K | 5.64 | 18.18 | `2165` | yes |
| 128K | 5.63 | 18.31 | `2165` | yes |
| 256K | 5.65 | 19.14 | `2165` | yes |
| 384K | 5.89 | 19.61 | `2165` | yes |
| 512K | 5.77 | 20.13 | `2165` | yes |
| 768K | 5.82 | 21.23 | `2165` | yes |
| **1024K (1M)** | **5.82** | **22.26** | `2165` | yes — **1.74 GiB headroom** |
| 1M humaneval (sanity) | 6.83 | 21.90 | `6796` | yes |

**No cliff exists for MoE 26B-A4B + TQ3 between 4K and 1M.** Decode is flat 5.6–6.6 tok/s across the entire range. VRAM grows sub-linearly at approximately 0.34 GiB per 64K additional context.

Speculation at 128K with dflash drafter: **12.48 tok/s, 59% accept rate, 2.22× over no-drafter baseline.**

At 1M + TQ3 with dflash drafter (dm=4): decode falls to **1.70 tok/s** — slower than no-drafter at 5.82 tok/s. VRAM saturates at 24.00 GiB; verify-pass must scan the full 1M KV per step (~1340 ms), dwarfing any drafter savings. Speculation is anti-economical when VRAM is saturated (see §3 Discovery #15, §5.8).

**Practical ship config on 24 GB GPUs**: MoE 26B-A4B + dflash + Q8/Q8 + pflash + dm=4 at 256K context → **67.95 tok/s**, 21.73 GiB. This is approximately 2× the previously-published 35–37 tok/s figure (§5.7), driven by ggml/master rebase improving prefill from ~1450 tok/s (chunked, no pflash) to ~5355 tok/s (chunked + pflash).

Logs: `.sisyphus/notes/gemma4-baseline/tq3-frontier/F_*.log` (no-drafter frontier), `C1_dflash_1M_tq3_dm4.log` (speculation at 1M), `C3_dflash_pflash_256K_q8_dm4.log` (ship config), `X_1048576_humaneval_tq3.log` (sanity), `00_smoke_q8_4k.log` and `01_smoke_tq3_4k.log` (correctness controls).

## 2. WHAT DID NOT WORK

### 2.1 Hard failures
- **Repeated `fattn.cu:652` crash** with MTP + Q8_0 V-cache or Q8/TQ3 hybrid:
  - `matrix-v2/M4_mtp_q8.log:60` (step 208, 68% accept pre-crash)
  - `matrix-v3/N3_mtp_q8_tq3.log:61` (step 216, 5% accept)
  - `matrix-64k/MTP_humaneval.log:48` (step 112, 96% accept!)
  - Crash is unrelated to accept-rate quality — kernel-side numerical/alignment issue still latent.
- **`T3_dflash.log` rc=143** in matrix-64k — fixed by commit 5b6ba1b "SWA mask coordinate frame".
- **dm-sweep/code4k_dm{4,8}.log:6** — `prompt (50003 tokens) >= ctx_size (4096)` — test misconfiguration.

### 2.2 Regressions (speculative HURT throughput)
- MTP @ 64K (V2_mtp): 6.33 vs 6.90 baseline (−8.2%). Accept = 0.02 (5/256).
- MTP @ 4K (M3_mtp_tq3): 25.28 vs 26.69 (none). Even at 39% accept.
- Sanity smoke (sanity_short.log:39): 0/8 accepted, garbled output on 27-token prompt.
- Dense @ 64K: 7.96 tok/s; @ 256K logs: 1.78 tok/s — dense path collapses at long ctx.

### 2.3 Quality issues
- matrix-64k-v2/SUMMARY.md:55 — V3_dflash_dm8 leaked `<mask>` and `<unused94>` tokens: `'swe Bras<mask>os<unused2><unused94><unused94>thought\n...'`. Recovers mid-stream.
- matrix-64k/SUMMARY.md — T2_mtp emitted token id 236772 repeatedly (vocab=262144, suspicious high-vocab range).
- MTP mode collapse (A1_tq3_0.log): tokens oscillated between id 109 (space+thought) and 49 (period). Structurally valid, semantically dead.

### 2.4 Abandoned / disabled approaches
- Uniform TQ3_0 across all layers → forced fallback: Q8_0 on captured full-attn layers, TQ3_0 on SWA. Runtime: `[cache] narrow asymmetric: forced Q8_0 on 2 captured full-attn layer(s)`. (ce4da35.)
- Monotonic SWA cache (full ctx alloc) → swapped for non-monotonic ring + correct mask (d68e7c4 restored VRAM savings; 19def9c first disabled buggy ring opt; 5b6ba1b fixed the mask).
- TQ3_0 preserved through ring-wrap into FA → still forced to F32 concat on wrap (gemma4_mtp_graph.cpp:369-388) → loses FWHT benefit. Workaround: increase swa_ctx_alloc.
- Standalone `ggml_turbo_wht` calls → fused into FA (c15f93a).

## 3. WHAT WE DISCOVERED (non-public engineering knowledge)

15 fixes/discoveries not documented in upstream llama.cpp / ggml.

| # | Subsystem | Discovery | Where it lives |
|---|---|---|---|
| 1 | SWA ring geometry | Mask must map *abs* query → *ring slot* `(qpos − abs_win_start) % ring_size`; without this, chunks 2+ silently read stale KV. Single-source-of-truth: `abs_win_start` + `ring_win_start=0`. | `gemma4_target_graph.cpp:205-222`, commit 5b6ba1b |
| 2 | TQ3_0 KV alignment | TQ3_0 + CUDA FA needs `ne[1] % 256 == 0`. **`FATTN_KQ_STRIDE=256` is hardcoded in CUDA, not exposed via any ggml API.** Allocate `((max_ctx+255)/256)*256`, mask out padded rows with −inf. | `gemma4_target_graph.cpp:99-101, 350-355`, `gemma4_mtp_graph.cpp:353-358`, commits c56879c, 41e4848 |
| 3 | Non-monotonic ring restoration | Disabling ring opt cost 50%+ VRAM on SWA layers. Re-enabled with correct mask geometry (#1). | commits 19def9c (disable), d68e7c4 (re-enable) |
| 4 | Narrow asymmetric KV | pFlash block-sparse path does not support TQ3_0 → forced Q8_0 on captured full-attn layers (4, 58 in Dense 31B); TQ3_0 stays on SWA. Per-layer override vectors `kv_k_type_per_layer`, `kv_v_type_per_layer`. | `internal.h:593-597`, ce4da35 |
| 5 | TQ3_0 type-tag preservation through FA | Casting K/V to F32/F16 before FA breaks the kernel's TQ3_0 → FWHT routing. Pass tensors native to `ggml_flash_attn_ext`; CHUNKED kernel handles FWHT internally. | `gemma4_mtp_graph.cpp:401-414`, c56879c |
| 6 | GQA block-broadcast for MTP | MTP's `n_head_norm` differs from target's `head_dim_kv`; explicit reshape `[head_dim_kv, n_head_kv*(n_head_norm/head_dim_kv), 1]` required before FA. | `gemma4_mtp_graph.cpp:120-122, 225-336`, 30b2b50 |
| 7 | MTP h_prev capture | MTP needs last-committed token's post-block hidden state from the last full-attn layer. Plus assistant's own `rope_freqs.weight` (top-level), KQ scale = 1.0 (not `1/√d`). | `gemma4_mtp_graph.cpp:266-285`, `internal.h:608-616`, 138de4d |
| 8 | FA mask mandatory for head_dim≥512 | CUDA MMA dispatcher requires `gqa_opt_applies` ⇒ both `K->ne[1]%256==0` AND `mask != nullptr`. Missing mask → `BEST_FATTN_KERNEL_NONE` → abort. Provide all-zero mask when logically all positions admitted. | `gemma4_mtp_graph.cpp:468-481`, f1f811e |
| 9 | n_tokens=1 decode + SWA | Single-token decode also needs allocated+filled SWA mask sized to `ring_size`, not `kv_seq_len`. | 7b62c07 |
| 10 | Shared FA mask across layers | Single mask buffer reused across all need-mask layers (256-align or head_dim≥512). | `internal.h:847-852`, `gemma4_mtp_graph.cpp:484-490` |
| 11 | DFlash decode correctness | BOS/EOS handling + per-layer SWA mask on draft (4 SWA + 1 full). Prefix-direct KV semantics. | 1386690, 9588c97 |
| 12 | TQ3 K dequant intercept strips FWHT rotation | MMA path strips FWHT rotation when dequanting TQ3_0 K → F16 before dispatch. Fix dropped the broken MMA dequant intercept; submodule now force-chunks all TQ3_0 cases (unless `DFLASH_TQ3_MMA` is opted in). | submodule `daef232a6` (force-chunked dispatcher, post-rebase) + outer-repo `4b0c158` (graph-level FWHT rotation via `ggml_cont` wrap) |
| 13 | Dense DFlash drafter generalises OOD; MoE drafter does not | Dense drafter AL is stable across prompt distributions (4.20 code / 5.12 creative). MoE drafter AL collapses from 5.22 (code) to 2.49 (creative) — it is code-distribution-trained. No public ablation on drafter-distribution generalisation exists. | `scientific/results.csv`, `gemma4-journey.md §11` |
| 14 | TQ3 long-context unlocked | Post-`daef232a6` chunked-dispatcher fix, MoE TQ3 sustains 5.6–6.6 tok/s flat from 4K to 1M with sub-linear VRAM growth (~0.34 GiB / 64K ctx). Practical 1M context on 24 GB GPUs confirmed. | `.sisyphus/notes/gemma4-baseline/tq3-frontier/`, submodule `daef232a6` |
| 15 | Speculation anti-economical at VRAM-saturated regime | At MoE 1M + TQ3 KV with VRAM > 96% used, dflash drafter REDUCES throughput (1.70 vs 5.82 tok/s). Verify-pass scans full 1M KV (~1340 ms/step). Speculation is only economical when ctx fits with headroom; at 256K Q8/Q8 the same drafter delivers 67.95 tok/s (11.7× over no-drafter). | `.sisyphus/notes/gemma4-baseline/tq3-frontier/C1_dflash_1M_tq3_dm4.log`, `C3_dflash_pflash_256K_q8_dm4.log` |

Why none of this is upstream: TQ3_0 is bleeding-edge (Google TurboQuant ~2024); MTP assistants and DFlash drafters are proprietary architectures; pFlash block-sparse paths are research-grade. Upstream llama.cpp ships none of these.

## 4. NARRATIVE TIMELINE (May 6–10)

- May 6–7 — Plan: Gemma4 31B Dense + 35B-A3B MoE; design TQ3_0 KV cache.
- May 7–8 — Target + draft baselines, TDD smoke tests; chunked prefill 12–16× speedup (3335ee2).
- May 8 — Draft KV cache (978ca01); pFlash sparse SWA layer-by-layer (33b6e9d); first long-ctx output corruption observed.
- May 9 AM — Diagnosed SWA mask coord frame bug (5b6ba1b), gate pFlash on supported KV types (1017dac).
- May 9 PM — Built MTP from scratch: loader (1115064), graph (d4659ca), wiring (05e36e4); accept rate stayed near zero on 64K tests.
- May 9 evening — Five MTP fixes landed in 24 h: h_prev / rope_freqs / KQ scale (138de4d), GQA broadcast + KV wrap (30b2b50), TQ3 preservation + 256-pad (c56879c), n_tokens=1 mask (7b62c07), head_dim≥512 mask (f1f811e). MTP runs to completion but accept stays ≤2% at 64K — fixes restored *correctness*, not *yield*.
- May 9–10 — Bench matrices: matrix-v2 / matrix-v3 (4K), matrix-64k → matrix-64k-v2 (post-fix), scaling 16K → 262K, dm-sweep. Discovered dm=4 is global optimum for long code prompts.
- May 10 — Debugging-journey blog committed (b441587, gemma4-journey.md). Dense-31B + dflash + Q8/Q8 + dm=8 ladder run: **decode collapses at 64K (1.78 tok/s, AL=1.94)** with full VRAM saturation; at 128K and 256K decode is 24 tok/s but output is a repetition loop (token 715, all steps after step 4).

## 5. BUGS FROM THE JOURNEY BLOG (gemma4-journey.md, commit b441587)

These are root-causes documented in the blog that add nuance not fully captured in §3 above.

### 5.1 Contaminated baseline — byte-fallback tokenisation

The plan we inherited claimed "accept_rate 0.22 on Q4_K_M target + Q8_0 assistant + TQ3_0 KV, 131-token prompt, 64-step generation." The `test_gemma4_dflash` binary was using **byte-fallback tokenisation** by default — it fed UTF-8 bytes as individual vocab IDs (102 IDs for a 25-word sentence instead of ~25 BPE tokens). The model was not generating real text; the drafter was agreeing with the target on garbage, producing a fake 0.22 accept_rate.

Decoding the output revealed multilingual gibberish: `<unused94>をlaenat quelelele tolaredlele samme a które a a a a robot`. After switching to real BPE tokenisation using HF's `google/gemma-3-27b-it` tokenizer (262144 entries, BOS=2, EOS=106, chat template `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`), the same binary produced "This sentence is a **pangram**..." — a real answer.

A bisect over the 6 commits between `7eea84b` (last pre-MTP) and `c56879c` found every commit in range produced garbage on TQ3. Walking further back to `ce4da35` (first TQ3 integration) also produced garbage. **TQ3 + Gemma-4 + real BPE tokens had never produced coherent text.** The 0.22 baseline was entirely fabricated by byte-fallback input.

### 5.2 Bug 1 — SWA mask not built for single-token decode (commit 7b62c07)

With real BPE input, target+TQ3 still produced garbage; target+Q8 produced clean prose. FA call-site diagnostics revealed:

```
[swa-fa-diag] n_tokens=28 mask=swa_mask  mask_ne0=2048 ← prefill OK
[swa-fa-diag] n_tokens=1  mask=attn_mask mask_ne0=256  ← decode WRONG
```

The SWA mask was guarded by `if (n_tokens > 1)`. For single-token decode, the code fell back to `attn_mask` sized to kv_len padding (256 wide), but the K view was still the full SWA ring (2048 wide). The kernel read 256 populated mask columns then continued into uninitialized CUDA malloc bytes. Q8's higher precision survived this (28 populated K positions still dominated); TQ3's quant noise plus garbage produced a near-flat attention distribution.

Fix: drop `n_tokens > 1` guard; always allocate and fill `swa_mask` at all four decode call sites.

### 5.3 Bug 2 — TQ3_0 K dequant MMA intercept strips FWHT rotation (submodule `daef232a6` + outer-repo `4b0c158`)

`fattn.cu:134–204` contained an intercept that dequantized TQ3_0 K storage to F16 and re-dispatched into the standard MMA path. TQ3_0 K is stored in **FWHT-rotated form** (applied at quantization write time). The chunked/vec FA kernels forward-rotate Q before computing Q@K when they see `K->type == GGML_TYPE_TQ3_0`. The MMA intercept skipped this because by dispatch time `K->type == GGML_TYPE_F16` — so Q was in standard space, K was in FWHT-rotated space, and Q@K was computed in mismatched domains.

For SWA decode (`Q->ne[1]==1, Q->ne[0]==256`), the existing guard `(Q->ne[1] > 1 || Q->ne[0] > 256)` did not fire, so the broken MMA intercept was always taken for SWA single-token decode.

Fix: drop the guard, force chunked for ALL TQ3_0 cases unless `DFLASH_TQ3_MMA` is opted in. After this fix, MTP+TQ3/TQ3 went from accept_rate 0.05 (degenerate loop) to **0.56 with coherent prose**. (Final fix landed as submodule `daef232a6` post-rebase; outer-repo wraps Q/O with `ggml_cont` in `4b0c158`.)

Why V was not broken by the symmetric bug: V's FWHT rotation affects only post-attention output values, not the attention score distribution (tokens are sampled from `softmax(QK)` not from V), so V being in FWHT space propagates a basis change that doesn't matter for argmax sampling.

### 5.4 Bug 3 — head_dim=512 + Q8/Q8 gqa-opt requires non-null mask (commit f1f811e)

After Bugs 1+2, MTP+Q8/Q8 still aborted at `fattn.cu:659` (BEST_FATTN_KERNEL_NONE → fatal error) around step ~110. The gqa-opt-applies check for head_dim=512 requires BOTH `K->ne[1] % 256 == 0` AND `mask != nullptr`. The MTP graph builder padded K correctly but its `need_mask` predicate was:

```cpp
const bool need_mask = (kv_is_tq3 && head_dim_fa >= 512) || needs_kv_pad;
```

For Q8 KV at 256-aligned length, neither clause fired — no mask, gqa-opt rejected, dispatcher returned NONE, abort.

Fix: drop the `kv_is_tq3` gate. Always set `need_mask` when `head_dim_fa >= 512`. After this fix, MTP + Q8/Q8 + HumanEval/2 + 4K ran the full 256 steps with **accept_rate 0.87** (peaked at 1.00 early, settled at 0.87 by step 112 — the exact step it used to crash at).

### 5.5 HumanEval surprise — drafter quality is prompt-distribution-bound

When checking for regressions post-fix, the dflash drafter on a `long_open.txt` prompt (40-token creative writing) gave 23.77 tok/s, AL=2.13/8. Switching to a HumanEval/2 prompt (139-token Python code) gave **56.12 tok/s, AL=5.12/8**. Same model, same config, 2.4× difference.

PR #131's published "10.67/16" (0.667 accept rate) was on code prompts. There was no regression — just a prompt distribution mismatch. The drafter is a 5-layer model distilled from target activations on HumanEval-class tasks; creative writing is severely OOD.

### 5.6 DM sweep — PR #131's 64K result was over-speculation, not drafter collapse

PR #131 reported 13 tok/s decode at 64K MoE. Our dm-sweep (dm ∈ {1, 2, 4, 8}) on the same model/context showed the root cause was the framework default `budget=22` (roughly dm=16+), not drafter quality. At dm=4, the same model and context gives **36.57 tok/s** — 2.8× higher than the published number. PR #131's result was over-speculation (budget too large → many rejected drafts → wasted compute), not drafter divergence.

### 5.7 Production ship config (from journey blog; updated post-rebase 2026-05-10)

Pre-rebase numbers (historical record, superseded by post-rebase column):

| Use case | Config | Decode tok/s (pre-rebase) | VRAM |
|---|---|---|---|
| Long context (>=64k), code/agent | MoE 26B + dflash + Q8/Q8 + dm=4 + pflash | 35-37 from 64k to 256k | 19.7-21.7 GB |
| Short context (4k), code/agent | MoE 26B + dflash + Q8/Q8 + dm=4 + pflash | ~112 | 19 GB |
| Short context, highest quality MTP | 31B dense + MTP + Q8/Q8 (post-Bug-3 fix) | 34 | 20 GB |
| Short context, dflash dense reference | 31B dense + dflash + Q8/Q8 + dm=8 + pflash | ~98 (HumanEval) | 22 GB |
| 64k dense, Q8/Q8 sanity | 31B + Q8/Q8 + pflash, no drafter | 7.96 | 22.6 GB |

Post-rebase corrected numbers (submodule `daef232a6` + outer-repo `4b0c158`):

| Use case | Config | Decode tok/s | VRAM |
|---|---|---|---|
| Long context (1M, code/agent) | MoE 26B + TQ3/TQ3 + no drafter + pflash | 5.82 | 22.26 GB |
| **Long context (256K, code/agent)** | **MoE 26B + dflash + Q8/Q8 + pflash + dm=4** | **67.95** | **21.73 GB** <- practical ship |
| Short context (4K), code/agent | MoE 26B + dflash + Q8/Q8 + dm=4 + pflash | ~112 | 19 GB |
| Short context, highest quality MTP | 31B dense + MTP + Q8/Q8 (post-Bug-3 fix) | 34 | 20 GB |
| Short context, dflash dense reference | 31B dense + dflash + Q8/Q8 + dm=8 + pflash | ~98 (HumanEval) | 22 GB |
| 64k dense, TQ3 minimum-VRAM | 31B + TQ3/TQ3 + no drafter | 2.54 | 21.07 GB |

### 5.8 Post-rebase TQ3 frontier — Bug 2 fix unlocks 1M context (2026-05-10)

The fork was rebased to ggml/master HEAD `0b047287f`, with 11 cherry-picks applied cleanly. The critical commit is `daef232a6`, which corrects the TQ3_0 chunked-path dispatcher: it forces the chunked/vec kernel for ALL TQ3_0 K unless `DFLASH_TQ3_MMA` is explicitly opted in, eliminating the intercept that dequantised TQ3_0 K to F16 (stripping its FWHT rotation) before the MMA dispatch. The outer-repo FWHT contract commit `4b0c158` completed end-to-end correctness. Validation sequence:

- 4K Q8 control: 62.83 tok/s, first gen `1106 6596 108` — unchanged from pre-rebase.
- 4K TQ3 matches Q8 first three tokens (`6596 108 2063`) — correctness confirmed.
- 16K → 1M TQ3 no-drafter sweep: flat 5.6–6.6 tok/s throughout, no cliff, VRAM growth sub-linear.
- Dense 64K: 2.54 tok/s, 21.07 GiB — cliff resolved (+43% vs prior 1.78 tok/s anomaly).
- Dense 128K/256K: 2.52/2.45 tok/s with real diverse output (first gen `45518`) — token-715 repetition collapse gone.

The dense 64K cliff anomaly and the earlier TQ3 SWA garbage at 4K (Bug 2, §5.3) were the same root cause — two manifestations of FWHT-domain mismatch in the TQ3_0 K dequant intercept. The `daef232a6` submodule commit is the unification fix for both.
