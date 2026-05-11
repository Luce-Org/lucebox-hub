# Phase 4-B — γ × ctx sweep results (approach B: multi-row h_prev)

Date: 2026-05-11 10:11–10:25 CEST
Binary: `dflash/build/test_gemma4_dflash` (post-Phase-3.5 build)
Approach: B — `mtp_h_prev_batch` `[n_embd, 17]` captures all K+1 rows in one verify call; host-side picks column `accept_drafts` after greedy match. No re-capture target forward.

## Setup

Identical to Phase 4-A: Dense 31B + assistant Q4_K_M, TQ3_0/TQ3_0 KV, `--temp 0 --ignore-eos --n-predict 64`, position mode const. Same three prompts.

## Decode tok/s — Phase 4-B vs Phase 4-A

|       | no-MTP        | γ=1   |       | γ=2   |       | γ=4   |       | γ=8   |       |
| ctx   | A      B      | A     | B     | A     | B     | A     | B     | A     | B     |
|-------|---------------|-------|-------|-------|-------|-------|-------|-------|-------|
| 4K    | 19.63   18.40 | 19.18 | 18.46 | 20.21 | 25.10 | 16.54 | **25.58** | 16.54 | 22.61 |
| 16K   | 12.99   12.43 | 10.20 |  9.79 |  8.40 | 12.56 |  5.37 | **13.06** |  6.86 |  9.33 |
| 64K   |  6.55    6.26 |  5.54 |  5.31 |  8.42 | **10.07** |  6.54 |  8.29 |  5.33 |  7.58 |

## Accept rate (drafts accepted / drafts proposed) — Phase 4-B

|       | γ=1   | γ=2   | γ=4   | γ=8   |
|-------|-------|-------|-------|-------|
| 4K    | 0.64  | 0.55  | 0.42  | 0.19  |
| 16K   | 0.64  | 0.47  | 0.47  | 0.23  |
| 64K   | 0.69  | 0.73  | 0.46  | 0.32  |

## Headline findings (approach B)

1. **γ=4 wins at 4K**: 25.58 tok/s = **+39% over no-MTP**. Up from +3% with approach A (γ=2 was the 4K winner there).
2. **16K "dead zone" is resolved**: γ=4 at 16K = 13.06 tok/s vs no-MTP 12.43 = **+5%**. Approach A's −35% loss was entirely the re-capture overhead.
3. **γ=2 at 64K is now +61% over no-MTP**: 10.07 vs 6.26 tok/s. Up from +29% with approach A.
4. **γ=8 still loses at 4K** (22.61 vs 25.58 at γ=4) and at 16K (9.33 vs 13.06). Diminishing returns — at γ=8 accept rate drops below 0.20–0.30 and the extra drafter steps cost more than they save.
5. **γ=1 regression-safe**: 18.46 tok/s vs M3 baseline 18.64 = −1% (noise). Accept rate identical at 0.66.

## Compared to approach A

|       | A best        | B best        | gain  |
|-------|---------------|---------------|-------|
| 4K    | γ=2 / 20.21   | γ=4 / 25.58   | +27%  |
| 16K   | no-MTP / 12.99| γ=4 / 13.06   | +0.5% (was −35% under A) |
| 64K   | γ=2 / 8.42    | γ=2 / 10.07   | +20%  |

The biggest unlock is at 16K: approach A made MTP unusable there; approach B makes it the right choice at every context.

## Updated user-facing defaults

| Ctx range | Best config                    | Decode tok/s | Note |
|-----------|--------------------------------|--------------|------|
| ≤8K       | `--draft-method mtp --gamma 4` | 25–26        | +39% over no-MTP |
| 8K–32K    | `--draft-method mtp --gamma 4` | 13           | +5% over no-MTP |
| 32K–256K  | `--draft-method mtp --gamma 2` | 10           | +61% at 64K |

## Open questions raised

1. Approach B's γ=4 wins at 4K and 16K but γ=2 still wins at 64K. The crossover is real — accept rate at 64K is high (0.73 for γ=2, 0.46 for γ=4), suggesting target's predictions diverge from drafter's at long range past depth 2. Worth a γ × ctx accept-rate plot.
2. γ=8 underperforms γ=4 everywhere. Drafter's autoregressive feedback at depth 8 drifts too far from target's hidden distribution. A retrained MTP head or tree-mask drafting (PR #22838) could push the sweet spot higher.
3. Need to test ≥128K to see if γ=2 trajectory continues. Memory should be fine — 64K used 21.78 GB peak.

## Logs

`{none_g0,mtp_g{1,2,4,8}}_ctx{4096,16384,65536}.log` in this directory.
Sweep script: `../run_sweep_b.sh`.
