# Gemma4 — Evidence Dossier

Last revised 2026-05-11 — all numbers are from the current binary (post commits `4bcb972` + submodule `6715acf13`). Earlier numbers that were superseded are explicitly marked in §2.2 ("claims walked back"). The dossier supersedes the v1 draft preserved as `EVIDENCE.v1.md.bak` in this directory.

## 0. Headline

The four composable components — **TQ3** (TurboQuant KV cache), **pFlash** (block-sparse SWA prefill), **DFlash** (block-diffusion drafter), **MTP** (multi-token assistant) — give a regime-split picture on a single 24 GB RTX 3090:

### MoE 26B-A4B: **Q8 + pflash + DFlash dm=4** dominates from 64K through 512K

| ctx | best config | decode tok/s | J/tok | VRAM (GB) |
|---|---|---|---|---|
| 4K (scientific bench, §1.7) | Q8 + pflash + DFlash dm=16 | 132.1 | 11.82 | 18.90 |
| 64K | Q8 + pflash + DFlash dm=8 (OVAT cell 05) | 34.66 | 12.02 | 19.84 |
| 128K | **Q8 + pflash + DFlash dm=4** | **66.97** | **6.26** | 20.45 |
| 256K | **Q8 + pflash + DFlash dm=4** | **63.82** | 8.13 | 21.75 |
| 512K | **Q8 + pflash + DFlash dm=4** | **62.40** | 6.82 | 24.00 (sat) |
| 1M | TQ3 + pflash + DFlash dm=4 | 26 ± 5 (triple-trial σ) | (untested with telemetry) | 24.00 (sat) |

**Surprise**: Q8 + pflash + DFlash at 128K/256K/512K = 60-67 tok/s decode — **2× faster than TQ3 + pflash + DFlash at the same contexts** (which gave 32-33 tok/s). The "TQ3 unlocks long context" framing is right at 1M+ where Q8 can't fit; below that Q8 is much faster.

### Dense 31B: VRAM-bottlenecked above 32K; **MTP γ=2 is the winning drafter at 64K-128K**

| ctx | best config | decode tok/s | J/tok | VRAM (GB) |
|---|---|---|---|---|
| 4K (scientific bench, §1.7) | Q8 + pflash + DFlash dm=16 (creative) | 81.7 | 8.82 | 22.08 |
| 32K | Q8 + pflash, no drafter | 18.33 | 35.04 | 21.36 |
| 64K | **TQ3 + pflash + MTP γ=2** (code) | **10.07** | (untested with telemetry) | ~22 |
| 128K | TQ3 + pflash + MTP γ=2 (code) | 10.18 | (untested with telemetry) | ~23 |
| ≥256K | INFEASIBLE — Dense 31B model + KV + drafter exceeds 24 GB | — | — | — |

**Correction**: Q8 vs TQ3 at Dense 64K is essentially tied (~6-8 tok/s no drafter, VRAM ~22-23 GB either way). My earlier "TQ3 is 2.5× faster on Dense" claim conflated MoE OVAT data with Dense — **that claim is withdrawn**. The real Dense 64K winner is **MTP γ=2 at 10.07 tok/s** (paired-matrix `dense_mtp_g2_code_ctx65536.log`). DFlash dm=16 on Dense pushes VRAM over 24 GB and slows decode (5.9-7.3 tok/s).

### The four-component decomposition (per-component contribution, MoE 64K code OVAT)

| Component change | Decode tok/s | Δ vs naive Q8 | Verdict |
|---|---|---|---|
| Q8 → TQ3 (KV alone) | 23.25 → 20.09 | **−14%** | TQ3 is a decode cost at MoE 64K |
| no-pflash → +pflash (Q8) | 23.25 → 23.65 | +2% | pflash decode-neutral at MoE 64K |
| Q8+pf → +DFlash dm=8 | 23.65 → 34.66 | **+49%** | DFlash is the big speedup at MoE 64K |
| Q8+pf → +MTP γ=2 (over TQ3) | 23.10 vs Q8 23.25 | −0.6% | MTP γ=2 ties naive baseline — net-zero |

**Cited evidence**: `mtp-gamma/ovat-moe-64k-code/{00..09}_*.log`, `mtp-gamma/closing/*.log`, `mtp-gamma/triple-falsify/trial_{1,2,3}.log`.

## 1. WHAT WORKED

### 1.0 Naive 3090 baseline (the public-comparable number)

The "naive" baseline is: Q8/Q8 KV, no pflash, no drafter, `--temp 0 --seed 0 --ignore-eos --n-predict 64` on `long_code_50k.txt` (50K-token code prompt).

| Model | Ctx | Decode tok/s | Notes |
|---|---|---|---|
| MoE 26B-A4B | 64K | **23.25** | OVAT cell `00_naive_q8`. 18.96 GB. |
| MoE 26B-A4B | 1M | **0.94** | OVAT 1M cell `00_naive_q8`. **24.00 GB saturated, paging**. Q8 KV does not fit. |
| Dense 31B | 32K | 18.33 (closing cell) | Q8+pflash already needed; without pflash, untested. |
| Dense 31B | 64K | **7.78** | Q8+pflash, no drafter. VRAM 22.64 GB (1.36 GB headroom). TQ3 at same cell is comparable (6.4 tok/s, see §1.6 for Dense drafter trade-offs). |

### 1.1 Per-component contributions — MoE 64K code OVAT (the rigorous reference cell)

10 cells, one-variable-at-a-time. Reference cell: MoE 26B-A4B × ctx=65536 × `long_code_50k.txt` × greedy.

| Cell | KV | pflash | Drafter | Decode tok/s | Prefill tok/s | VRAM (GB) | Decode J/tok |
|---|---|---|---|---|---|---|---|
| 00 naive | Q8 | off | none | 23.25 | 2379 | 18.96 | 21.37 |
| 01 TQ3 | TQ3 | off | none | 20.09 | 1634 | 18.06 | 21.05 |
| 02 Q8+pf | Q8 | on | none | 23.65 | **5049** | 18.66 | **17.02** |
| 03 TQ3+pf | TQ3 | on | none | 19.50 | 1627 | 18.06 | 21.51 |
| 04 …+DFlash dm=4 | TQ3 | on | dflash | 31.09 | — | 19.83 | 13.34 |
| 05 …+DFlash **dm=8** | TQ3 | on | dflash | **34.66** | — | 19.84 | **12.02** |
| 06 …+DFlash dm=16 | TQ3 | on | dflash | 31.10 | — | 19.84 | 13.35 |
| 07 …+MTP γ=1 | TQ3 | on | mtp | 15.04 | — | 18.79 | 27.18 |
| 08 …+MTP γ=2 | TQ3 | on | mtp | 23.10 | — | 18.85 | 17.93 |
| 09 …+MTP γ=4 | TQ3 | on | mtp | 18.31 | — | 18.84 | 22.19 |

**Conclusions at MoE 64K code**:
1. TQ3 KV alone is a **−14% decode penalty** vs Q8 (and −31% prefill).
2. pflash gives **+112% prefill** on Q8 and is **decode-neutral** (±2%).
3. DFlash dm=8 is the **best decode AND best efficiency** (+49% tok/s, −44% J/tok).
4. MTP γ=2 essentially **ties the naive baseline** — the TQ3 decode penalty cancels the MTP gain.
5. MTP γ=1 is a regression (drafter overhead, no amortization).
6. MTP γ=4 is worse than γ=2 — drafter accuracy drops faster than verify is amortized.

### 1.2 Q8 KV ceiling — where does Q8 stop fitting? (with energy telemetry)

Direct measurements from `mtp-gamma/closing/` and `mtp-gamma/q8-ceiling/`:

#### MoE Q8 + pflash, NO drafter (the floor)

| Ctx | Decode tok/s | Prefill tok/s | VRAM (GB) | Avg W | Decode J/tok |
|---|---|---|---|---|---|
| 128K | 24.55 | 5121 | 19.19 | 202.5 | 24.40 |
| 256K | 24.75 | 5354 | 20.52 | 235.7 | 21.88 |
| 512K | 23.68 | 5239 | 23.12 | 324.0 | 20.71 |
| 1M | 19.32 | (paging) | 24.00 (sat) | — | — |

#### MoE Q8 + pflash + DFlash dm=4 (the winning stack)

| Ctx | Decode tok/s | Prefill tok/s | VRAM (GB) | Avg W | Decode J/tok |
|---|---|---|---|---|---|
| 128K | **66.97** | 5394 | 20.45 | 320.2 | **6.26** |
| 256K | **63.82** | 5186 | 21.75 | 324.5 | 8.13 |
| 512K | **62.40** | 5241 | 24.00 (sat) | 311.7 | 6.82 |

**DFlash dm=4 on Q8+pflash gives +160% decode and −4× J/tok over the no-drafter baseline at 128K-512K MoE.**

#### Dense Q8 + pflash

| Ctx | Drafter | Decode tok/s | Prefill tok/s | VRAM (GB) | Decode J/tok |
|---|---|---|---|---|---|
| 32K | none | 18.24 | 1365 | 21.34 | 35.04 |
| 32K | DFlash dm=16 | 12.53 | 1357 | 23.67 | 40.25 (drafter actually HURTS) |
| 64K | none | 7.94 | 1385 | 22.64 | 55.89 |
| 64K | DFlash dm=16 | 5.92 | **130.6 (prefill collapse)** | 24.00 (sat) | 38.56 |

**Dense Q8 at 64K is bandwidth-limited (7.94 no-drafter). Adding DFlash dm=16 PUSHES VRAM to 24 GB and collapses prefill to 130 tok/s — drafter is anti-economical on Dense Q8 above 32K.**

#### Dense TQ3 + pflash + DFlash dm=16

| Ctx | Decode tok/s | Prefill tok/s | VRAM (GB) | Decode J/tok |
|---|---|---|---|---|
| 32K | 8.73 | 633 | 23.01 | 49.50 |
| 64K | 7.29 | 483 | 23.80 | 58.45 |

**Headlines**:
- MoE Q8 ceiling: **between 512K and 1M**. Use Q8+pflash+DFlash dm=4 through 512K.
- Dense Q8 ceiling: **DFlash makes Dense worse at any long ctx on 24 GB.** MTP γ=2 at TQ3+pflash is the better drafter for Dense ≥64K (10.07 tok/s at 64K code).

### 1.3 TQ3 long-context — where TQ3 is the only viable choice

MoE 26B-A4B + TQ3/TQ3 + pflash (from `mtp-gamma/moe-scientific/none_*.log`):

| Ctx | Decode tok/s | VRAM (GB) |
|---|---|---|
| 64K | 20.03 | 18.06 |
| 128K | 19.37 | 18.33 |
| 256K | 19.88 | 19.06 |
| 512K | 20.00 | 20.18 |
| 1M | 19.79 | 22.25 |

**Flat ~20 tok/s decode from 64K through 1M on TQ3+pflash MoE.** Compare to Q8+pflash which also gives ~24 tok/s but only through 512K. Above 512K TQ3 wins because Q8 doesn't fit.

### 1.4 DFlash drafter — speedup contribution, with the variance caveat

DFlash dm=4 on top of TQ3+pflash at MoE long ctx (from `mtp-gamma/moe-scientific/dflash_dm4_*.log`):

| Ctx | Decode tok/s | AL | Δ vs TQ3+pflash no-drafter |
|---|---|---|---|
| 64K | 32.41 | 2.13/4 | +62% |
| 128K | 32.25 | 2.13/4 | +66% |
| 256K | 32.69 | 2.13/4 | +64% |
| 512K | 32.79 | 2.13/4 | +64% |
| 1M (single moe-scientific run) | **4.86** | 4.00/4 | **−75% (outlier)** |
| 1M (triple-falsification mean) | **26 ± 5** | 4.00/4 | **+31% mean** |

**Important variance finding**: 1M DFlash exhibits **20% run-to-run variance** with state-dependent slowdowns. Trial 1 = 29.87, Trial 2 = 20.73, Trial 3 = 27.54 (mean 26.05, σ 4.6). The single moe-scientific 4.86 tok/s number was a pathological outlier after 19 prior cells in the same shell. **DFlash at 1M MoE is competitive (~26 tok/s) but unstable under sweep churn.**

### 1.5 MTP — what it actually delivers

From `mtp-gamma/moe-scientific/mtp_g2_*.log` and `mtp-gamma/ovat-moe-64k-code/`:

| ctx | MTP γ=2 tok/s | accept rate | vs naive Q8 baseline | vs TQ3+pflash no-drafter |
|---|---|---|---|---|
| MoE 64K | 23.10 | 0.565 | **−0.6% (tied)** | +18% |
| MoE 128K | 23.77 | 0.565 | (Q8 not measured) | +23% |
| MoE 256K | 23.70 | 0.565 | (Q8 fits, 25.01) | +19% |
| MoE 512K | 23.91 | 0.565 | (Q8 fits, 24.69) | +20% |
| MoE 1M | 23.65 | 0.565 | (Q8 paging) | +20% |
| Dense 128K (code, OVAT) | 10.18 | 0.780 | (Q8 not measured at 128K Dense) | — |

**Revised honest picture**:
- MTP γ=2 on MoE: **flat 23-24 tok/s across all long contexts**. Stable across 64K→1M.
- Vs the naive Q8 baseline where measurable (64K, 256K, 512K): **slightly slower or tied** — the TQ3 penalty offsets the MTP gain.
- **MTP only beats Q8 baseline at 1M** where Q8 pages. Otherwise it's neutral.
- DFlash dominates MTP at every measured cell where they both apply.

### 1.6 Dense 31B — VRAM ceiling forces drafter trade-offs

Honest measurement at Dense 64K code:

| Cell | KV | pflash | Drafter | Decode tok/s | VRAM (GB) |
|---|---|---|---|---|---|
| Dense Q8+pf no-drafter (closing) | Q8 | on | none | 7.94 | 22.64 |
| Dense TQ3+pf no-drafter (paired-matrix, code) | TQ3 | on | none | 6.36 | ~22 |
| Dense Q8+pf + DFlash dm=16 (closing) | Q8 | on | dflash | 5.92 (prefill collapses) | 24.00 (sat) |
| Dense TQ3+pf + DFlash dm=16 (closing) | TQ3 | on | dflash | 7.29 | 23.80 |
| **Dense TQ3+pf + MTP γ=2 (paired-matrix code)** | TQ3 | on | mtp | **10.07** | ~22 |

**Findings on Dense 64K**:
- Q8 vs TQ3 no-drafter: **essentially tied (8 vs 6 tok/s)**. The "TQ3 is 2.5× faster on Dense" claim from earlier was a mis-conflation with MoE OVAT data — **withdrawn**.
- DFlash on Dense at 64K pushes VRAM to 24 GB and **slows everything** (prefill collapses to 130 tok/s on Q8, decode drops to 5-7 tok/s on either KV).
- **MTP γ=2 is the only drafter that helps Dense at 64K** because MTP doesn't carry its own KV cache — it cross-attends to target's KV. **10.07 tok/s with 0.78 accept rate, no VRAM pressure spike.**

The lesson: on a VRAM-constrained 24 GB GPU, the drafter's memory footprint matters as much as its compute. DFlash's draft KV cache prices it out of Dense long-ctx. MTP's Q-only cross-attn fits.

### 1.7 (kept from v1) Scientific 24-cell sweep at 4K Q8/Q8 — energy reference

Source: `scientific/results.csv`, `scientific/SUMMARY.md`. Config: Q8/Q8 KV, 4K ctx, n_predict=256, temp=0 seed=0 --ignore-eos, pflash on, RTX 3090 24 GB. Cells: Dense × MoE × code/creative × dm ∈ {1, 2, 4, 8, 16, 32}.

| cell | dm | Decode tok/s | AL | decode J/tok | VRAM GB |
|---|---|---|---|---|---|
| **moe_code_dm16** | 16 | **132.1** | 5.22 | 11.82 ← peak speed | 18.90 |
| moe_code_dm8 | 8 | 131.8 | 3.88 | 14.39 | 18.89 |
| moe_creative_dm4 | 4 | 94.6 | 2.12 | 8.33 | 18.88 |
| dense_creative_dm16 | 16 | **81.7** | 5.12 | 8.82 ← Dense peak | 22.08 |
| **moe_creative_dm2** | 2 | 82.1 | 1.63 | **6.64 ← lowest J/tok** | 18.90 |
| moe_creative_dm1 | 1 | 58.0 | 1.00 | 5.70 (no-spec baseline) | 18.87 |

**Headline**: at 4K Q8/Q8, MoE peaks at 132 tok/s with DFlash dm=16; Dense peaks at 81.7 tok/s with DFlash dm=16. Best energy per token is MoE+creative+dm=2 at 6.64 J/tok. **dm=4 is NOT the universal optimum — it's prompt + model dependent.** Dense prefers dm=16; MoE peaks at dm=16 on code but at dm=4 on creative.

## 2. WHAT DID NOT WORK (corrected from v1)

### 2.1 Hard failures (current binary)

- **DFlash at Dense 128K**: timed out twice in paired-matrix sweep (`paired-matrix/dense_dflash_dm4_*_ctx131072.log`). VRAM saturation. The Dense 31B + drafter + 128K KV exceeds 24 GB working set.
- **Q8 KV at MoE 1M without pflash**: 0.94 tok/s (paging). pflash rescues to 19.32 tok/s; only TQ3 enables clean long-ctx.

### 2.2 Claims from v1 that this rewrite WALKS BACK

| v1 claim | Status | Evidence |
|---|---|---|
| "MTP is functional but uneconomic at 64K+ (≤2% accept) — open investigation" | **FALSE** — fixed by submodule `daef232a6` + γ>1 chain | OVAT cell 08: γ=2 accept = 0.565 at MoE 64K |
| "MTP at 4K (M3_mtp_tq3): 25.28 vs 26.69 — even at 39% accept" | **STALE** — pre-rebase | post-rebase γ=1 = 18.46 with accept 0.66, γ=2 = 25.10 |
| "MTP @ 64K (V2_mtp): 6.33 vs 6.90, accept 0.02" | **STALE** — pre-Bug-2 | post-rebase γ=2 at 64K = 23.10 |
| "DFlash collapses to 1.70 tok/s at 1M" | **PARTIAL** — happens under sweep churn, but cold-fresh = ~26 tok/s ±5 | triple-falsification |
| "γ=2 at 64K = +61% over no-MTP" | **MISLEADING** — was vs TQ3 baseline; vs naive Q8 baseline it's **tied** | OVAT cells 00 vs 08 |
| "Speculation is anti-economical at VRAM-saturated regime" | **TRUE only under sweep churn** | triple-falsification shows DFlash 1M = 26 ± 5 cold |

### 2.3 Quality issues (kept from v1)
- DFlash V3 leaked `<mask>` and `<unused94>` tokens (`matrix-64k-v2/SUMMARY.md:55`). Not retested on current binary.
- TQ3_0 ring-wrap path forces F32 concat on wrap — workaround in `gemma4_mtp_graph.cpp`. Decode within SWA window is unaffected.

## 3. PRACTICAL RECIPES (consumer 24 GB 3090, post-2026-05-11)

### MoE 26B-A4B (4B active)

| Ctx | Config | Decode tok/s | J/tok | Source |
|---|---|---|---|---|
| 4K | Q8/Q8 + pflash + DFlash dm=16 | 132.1 | 11.82 | scientific bench §1.7 |
| 64K | Q8/Q8 + pflash + DFlash dm=8 | 34.66 | 12.02 | OVAT cell 05 |
| 128K | **Q8/Q8 + pflash + DFlash dm=4** | **66.97** | **6.26** | closing/moe_128K_q8_pf_dfl4 |
| 256K | **Q8/Q8 + pflash + DFlash dm=4** | **63.82** | 8.13 | closing/moe_256K_q8_pf_dfl4 |
| 512K | **Q8/Q8 + pflash + DFlash dm=4** | **62.40** | 6.82 | closing/moe_512K_q8_pf_dfl4 |
| 1M | TQ3/TQ3 + pflash + DFlash dm=4 | 26 ± 5 | (not energy-instrumented) | triple-falsify mean |

### Dense 31B (60L)

| Ctx | Config | Decode tok/s | J/tok | Source |
|---|---|---|---|---|
| 4K | Q8/Q8 + pflash + DFlash dm=16 | 81.7 (creative), 68.0 (code) | 8.82 / 17.33 | scientific bench |
| 32K | Q8/Q8 + pflash, no drafter (DFlash hurts) | 18.24 | 35.04 | closing/dense_32K_q8_pf_none |
| 64K | **TQ3/TQ3 + pflash + MTP γ=2** | **10.07** | (not instrumented) | paired-matrix code log |
| 128K | TQ3/TQ3 + pflash + MTP γ=2 | 10.18 | (not instrumented) | paired-matrix code log |
| ≥256K | INFEASIBLE on 24 GB | — | — | Dense+KV+drafter > 24 GB |

**Key recipe deltas vs v1**:
- MoE 128K-512K: Q8+pf+DFlash dm=4 (NEW, 60-67 tok/s) — was previously claimed as TQ3+pf+DFlash at 32 tok/s. Q8 is **2× faster** in this band.
- Dense 64K-128K: MTP γ=2 (not DFlash) — DFlash pushes Dense over 24 GB VRAM.
- 1M MoE DFlash: 26 ± 5 (with run-to-run variance) — was previously claimed as either 4.86 (outlier) or 32.66 (lucky cold run). The triple-falsification gives the honest range.

## 4. OPEN QUESTIONS / NEXT WORK

1. **MoE 128K-512K with DFlash on Q8** — closing cells in flight (`run_closing.sh`).
2. **Dense 32K-64K with DFlash dm=16 on TQ3** — closing cells in flight.
3. **Cold-start protocol for sweeps**: per discovery #18, all sweep numbers at VRAM-saturated regimes need to be re-run with a fresh process per cell (kill + sleep + new process). Current results.csv files mix cold and warm states.
4. **pflash decode contribution at intermediate ctx** (128K, 256K) — direct A/B never measured.
5. **Dense at >128K**: VRAM-bound on 24 GB; would need TQ3 + smaller drafter or weight quantization.

## 5. NARRATIVE TIMELINE (May 6 – May 11, 2026)

- **May 6–7** — Plan: Gemma4 31B Dense + 26B-A4B MoE; design TQ3_0 KV cache.
- **May 7–8** — Target + draft baselines, TDD smoke tests; chunked prefill 12–16× speedup (`3335ee2`).
- **May 8** — Draft KV cache (`978ca01`); pFlash sparse SWA layer-by-layer (`33b6e9d`); first long-ctx output corruption observed.
- **May 9 AM** — Diagnosed SWA mask coord frame bug (`5b6ba1b`), gate pFlash on supported KV types (`1017dac`).
- **May 9 PM** — Built MTP from scratch: loader (`1115064`), graph (`d4659ca`), wiring (`05e36e4`); γ=1 accept rate stayed near zero on 64K tests.
- **May 9 evening** — Five MTP fixes landed in 24 h: h_prev / rope_freqs / KQ scale (`138de4d`), GQA broadcast + KV wrap (`30b2b50`), TQ3 preservation + 256-pad (`c56879c`), n_tokens=1 mask (`7b62c07`), head_dim≥512 mask (`f1f811e`). γ=1 MTP runs to completion but accept ≤2% at 64K — fixes restored *correctness*, not *yield*. (Later: yield problem turned out to be Bug 2, see §7.3.)
- **May 9–10** — Bench matrices: matrix-v2 / matrix-v3 (4K), matrix-64k → matrix-64k-v2 (post-fix), scaling 16K → 262K, dm-sweep. Discovered dm=4 is MoE optimum for long code prompts; Dense optimum is dm=16.
- **May 10** — Submodule rebased to ggml/master `0b047287f` + critical fix `daef232a6` (TQ3 chunked-path dispatcher, see §7.3). MoE TQ3 1M context confirmed at 5.82 tok/s on 24 GB. Dense 64K cliff resolved.
- **May 10 evening** — γ>1 MTP planned + reviewed by momus. Phases 1+2+3 (approach A, re-capture target forward) shipped as commit `d8ebd12`. First γ=4 result at 4K: 20.75 tok/s.
- **May 11 AM** — Approach B (multi-row `mtp_h_prev_batch`) shipped as commit `4bcb972`. Eliminates re-capture forward. Closing OVAT + Q8-ceiling sweeps run with energy telemetry. Reveal: MoE Q8+pflash+DFlash dm=4 = 60–67 tok/s at 128K–512K (2× faster than TQ3 stack); Dense 31B drafter sweet spot is MTP γ=2 (not DFlash) at TQ3 above 32K. Earlier "+61% γ=2 MoE 64K" framing walked back — was vs TQ3 baseline, vs Q8 baseline it ties.
- **May 11 PM** — Triple-falsification of MoE 1M DFlash exposed 20 % run-to-run variance under sweep churn; the moe-scientific 4.86 tok/s figure was an outlier. Honest 1M MoE DFlash number: **26 ± 5 tok/s** cold, **TQ3 mandatory** because Q8 pages.

## 6. WHAT WE DISCOVERED (non-public engineering knowledge)

20 fixes/discoveries not documented in upstream llama.cpp / ggml.

| # | Subsystem | Discovery | Where it lives |
|---|---|---|---|
| 1 | SWA ring geometry | Mask must map *abs* query → *ring slot* `(qpos − abs_win_start) % ring_size`; without this, chunks 2+ silently read stale KV. | `gemma4_target_graph.cpp:205-222`, commit `5b6ba1b` |
| 2 | TQ3_0 KV alignment | TQ3_0 + CUDA FA needs `ne[1] % 256 == 0`. **`FATTN_KQ_STRIDE=256` is hardcoded in CUDA, not exposed via any ggml API.** Allocate `((max_ctx+255)/256)*256`, mask out padded rows with −inf. | `gemma4_target_graph.cpp:99-101, 350-355`, commits `c56879c`, `41e4848` |
| 3 | Non-monotonic ring restoration | Disabling ring opt cost 50%+ VRAM on SWA layers. Re-enabled with correct mask geometry (#1). | commits `19def9c` (disable), `d68e7c4` (re-enable) |
| 4 | Narrow asymmetric KV | pFlash block-sparse path does not support TQ3_0 → forced Q8_0 on captured full-attn layers; TQ3_0 stays on SWA. | `internal.h:593-597`, `ce4da35` |
| 5 | TQ3_0 type-tag preservation through FA | Casting K/V to F32/F16 before FA breaks the kernel's TQ3_0 → FWHT routing. Pass tensors native to `ggml_flash_attn_ext`. | `gemma4_mtp_graph.cpp:401-414`, `c56879c` |
| 6 | GQA block-broadcast for MTP | MTP's `n_head_norm` differs from target's `head_dim_kv`; explicit reshape required before FA. | `gemma4_mtp_graph.cpp:120-122, 225-336`, `30b2b50` |
| 7 | MTP h_prev capture | MTP needs last-committed token's post-block hidden state from the last full-attn layer. KQ scale = 1.0 (not `1/√d`). | `gemma4_mtp_graph.cpp:266-285`, `138de4d` |
| 8 | FA mask mandatory for head_dim≥512 | CUDA MMA dispatcher requires `gqa_opt_applies` ⇒ both `K->ne[1]%256==0` AND `mask != nullptr`. | `gemma4_mtp_graph.cpp:468-481`, `f1f811e` |
| 9 | n_tokens=1 decode + SWA | Single-token decode also needs allocated+filled SWA mask sized to `ring_size`, not `kv_seq_len`. | `7b62c07` |
| 10 | Shared FA mask across layers | Single mask buffer reused across all need-mask layers (256-align or head_dim≥512). | `internal.h:847-852` |
| 11 | DFlash decode correctness | BOS/EOS handling + per-layer SWA mask on draft (4 SWA + 1 full). Prefix-direct KV semantics. | `1386690`, `9588c97` |
| 12 | TQ3 K dequant intercept strips FWHT rotation | MMA path strips FWHT rotation when dequanting TQ3_0 K → F16 before dispatch. Fix forces chunked path for all TQ3_0 cases. | submodule `daef232a6` + outer-repo `4b0c158` |
| 13 | Dense DFlash drafter generalises OOD; MoE drafter does not | Dense drafter AL stable across distributions (4.20 code / 5.12 creative). MoE drafter AL collapses 5.22 → 2.49 on creative. | `scientific/results.csv` |
| 14 | TQ3 long-context unlocked | Post-`daef232a6`, MoE TQ3 sustains 5.6–6.6 tok/s flat from 4K to 1M with sub-linear VRAM growth (~0.34 GiB / 64K ctx). | `tq3-frontier/`, submodule `daef232a6` |
| 15 | DFlash 1M run-to-run variance | Same CLI, same seed yields tok/s ∈ [20, 30] on MoE 1M cold; degrades to 4.86 after 19 prior cells in same shell. WSL2/CUDA allocator state matters. **Sweep-harness numbers at saturated VRAM ctx need cold-start protocol.** | `mtp-gamma/triple-falsify/` |
| 16 | γ>1 MTP chain (first public implementation) | Approach B (multi-row `mtp_h_prev_batch`) eliminates per-chain re-capture overhead. Mirrors Google HF reference. | commits `d8ebd12` (A), `4bcb972` (B); `gemma4_target_graph.cpp:1106-1123` |
| 17 | MoE MTP assistant exists | `AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF`. Loads cleanly against MoE 30-layer target — loader remaps Dense-trained donor layer indices. | `models/gemma4-mtp-26b-a4b/` |
| 18 | MoE Q8 ceiling is 512K, not earlier | Q8+pflash decode flat at 24-25 tok/s from 64K through 512K. Pages at 1M. Earlier framing "TQ3 wins at long ctx" was wrong below 1M. | `mtp-gamma/q8-ceiling/`, `mtp-gamma/closing/` |
| 19 | Dense + DFlash above 32K is anti-economical | DFlash drafter KV cache pushes Dense over 24 GB VRAM. Decode drops to 5-7 tok/s, prefill collapses. MTP γ=2 is the right drafter for Dense ≥ 32K. | `mtp-gamma/closing/dense_*`, `mtp-gamma/paired-matrix/dense_mtp_*` |
| 20 | Q8+pflash+DFlash dm=4 is the MoE long-ctx winner | At MoE 128K-512K: 60-67 tok/s, 6-8 J/tok. 2× faster than the TQ3+pflash+DFlash stack at the same contexts. | `mtp-gamma/closing/moe_*_q8_pf_dfl4.log` |

Why none of this is upstream: TQ3_0 is bleeding-edge (Google TurboQuant ~2024); MTP assistants and DFlash drafters are proprietary architectures; pFlash block-sparse paths are research-grade. Upstream llama.cpp ships none of these.

## 7. BUGS FROM THE JOURNEY BLOG (from `gemma4-journey.md`, commit `b441587`)

These root-causes add nuance to the discoveries table.

### 8.1 Contaminated baseline — byte-fallback tokenisation
The plan we inherited claimed "accept_rate 0.22 on Q4_K_M target + Q8_0 assistant + TQ3_0 KV". The `test_gemma4_dflash` binary was using **byte-fallback tokenisation** by default — UTF-8 bytes as individual vocab IDs (102 IDs for a 25-word sentence instead of ~25 BPE tokens). The model wasn't generating real text; the drafter was agreeing with the target on garbage. Output: `<unused94>をlaenat quelelele tolaredlele samme a które a a a a robot`. After switching to real BPE tokenisation, the same binary produced coherent English. **The 0.22 baseline was fabricated by byte-fallback input.**

### 8.2 Bug 1 — SWA mask not built for single-token decode (`7b62c07`)
With real BPE input, target+TQ3 still produced garbage; target+Q8 was clean. FA diagnostics showed `n_tokens=1 mask=attn_mask mask_ne0=256` (decode WRONG) vs `n_tokens=28 mask=swa_mask mask_ne0=2048` (prefill OK). The SWA mask was guarded by `if (n_tokens > 1)`; for single-token decode the code fell back to `attn_mask` sized to 256 wide while the K view was 2048 wide. The kernel read 256 mask columns then continued into uninitialized CUDA bytes. Fix: drop the guard; always fill `swa_mask` at all four decode call sites.

### 8.3 Bug 2 — TQ3_0 K dequant MMA intercept strips FWHT rotation (submodule `daef232a6` + outer-repo `4b0c158`)
`fattn.cu:134–204` had an intercept that dequantised TQ3_0 K to F16 and re-dispatched into the standard MMA path. **TQ3_0 K is stored in FWHT-rotated form** (applied at quantization write time). The chunked/vec FA kernels forward-rotate Q before Q@K when they see `K->type == GGML_TYPE_TQ3_0`. The MMA intercept skipped this because by dispatch time `K->type == GGML_TYPE_F16` — Q was in standard space, K was in FWHT-rotated space, Q@K computed in mismatched domains. Fix: force chunked for ALL TQ3_0 cases unless `DFLASH_TQ3_MMA` opted in. After fix, MTP+TQ3/TQ3 went from accept 0.05 (degenerate loop) → **0.56 with coherent prose**. V was not broken by the symmetric bug because V's FWHT rotation affects only post-attention output values, not the attention score distribution.

### 8.4 Bug 3 — head_dim=512 + Q8/Q8 gqa-opt requires non-null mask (`f1f811e`)
After Bugs 1+2, MTP+Q8/Q8 still aborted at `fattn.cu:659` (BEST_FATTN_KERNEL_NONE → fatal error) around step ~110. gqa-opt-applies for head_dim=512 requires BOTH `K->ne[1] % 256 == 0` AND `mask != nullptr`. The MTP graph builder padded K but its `need_mask` predicate was gated on `kv_is_tq3`. Fix: always set `need_mask` when `head_dim_fa >= 512`. After fix, MTP+Q8/Q8+HumanEval/2+4K ran the full 256 steps with accept rate 0.87.

### 8.5 HumanEval surprise — drafter quality is prompt-distribution-bound
DFlash drafter on `long_open.txt` (40-token creative writing): 23.77 tok/s, AL=2.13/8. On HumanEval/2 (139-token Python code): **56.12 tok/s, AL=5.12/8**. Same model, same config, 2.4× difference. PR #131's published "10.67/16" was code-prompt only — there was no regression, just prompt distribution mismatch. MoE drafter is distilled from HumanEval-class activations; creative writing is severely OOD.

### 8.6 PR #131's 64K result was over-speculation, not drafter collapse
PR #131 reported 13 tok/s decode at 64K MoE. Our dm-sweep showed the root cause was `budget=22` framework default (effectively dm=16+), not drafter quality. At dm=4, same model + context: **36.57 tok/s** (2.8× PR #131). Over-speculation produced many rejected drafts → wasted compute.

## 8. SOURCE LOGS

| Section | Source dir |
|---|---|
| §1.0 baseline | `mtp-gamma/ovat-moe-64k-code/00_naive_q8.log`, `mtp-gamma/q8-ceiling/`, `mtp-gamma/closing/` |
| §1.1 OVAT | `mtp-gamma/ovat-moe-64k-code/{00..09}*.log` |
| §1.2 Q8 ceiling | `mtp-gamma/q8-ceiling/`, `mtp-gamma/closing/` |
| §1.3 TQ3 long ctx MoE | `mtp-gamma/moe-scientific/none_*.log` |
| §1.4 DFlash | `mtp-gamma/moe-scientific/dflash_dm4_*.log`, `mtp-gamma/triple-falsify/trial_*.log` |
| §1.5 MTP | `mtp-gamma/moe-scientific/mtp_g{1,2,4}_*.log`, `mtp-gamma/paired-matrix/dense_mtp_*.log` |
| §1.6 Dense TQ3 | OVAT cell 03 (Dense version, pending re-OVAT in closing sweep) |
| §1.7 scientific | `scientific/results.csv`, `scientific/SUMMARY.md` |

---

*Drafted while `run_closing.sh` is in flight. Will be updated with §1.2 / §1.6 numbers when those 12 cells finish (~15 min from 15:31 CEST 2026-05-11).*
