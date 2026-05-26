# gemma-4-26b thinking-budget sweep — aime2025-02 — 2026-05-26

Single-case probe: a hard AIME geometry problem (correct answer = **588**)
at varied `thinking.budget_tokens` to find the diminishing-returns
threshold.

Server: bragi luce-dflash (cuda12 image), gemma-4-26b-a4b-it Q4_K_M,
temp=1.0/top_p=0.95/top_k=64, hard_limit_reply_budget=4096
(so `comp = budget + 4096` when force-close fires).

| budget_tokens | comp tokens | wall (s) | decode_tps | got 588? | final answer |
|---|---|---|---|---|---|
| 512   | 4608  | 44.3  | 105 | derivation only | 1 ❌ |
| 1024  | 4042  | 39.4  | 103 | NO              | 420 ❌ |
| 2048  | 4693  | 46.3  | 102 | YES             | **588 ✓** |
| 4096  | 5249  | 52.2  | 101 | YES             | **588 ✓** |
| 8192  | 9489  | 94.4  | 101 | YES             | **588 ✓** |
| 16384 | 13468 | 133.5 | 101 | YES             | **588 ✓** |

**Sweet spot ≈ 2048**. Below it the model gets force-closed before
reaching the answer (the post-close reply-budget can't recover —
content is "Final Answer: 1"). At or above 2048 the model self-closes
naturally and consistently lands at 588. Going from 2048 → 16384
costs 2.9× more wall (46s → 134s) for zero quality gain.

Caveats:
- Single AIME case (geometry, ground-truth integer answer).
- A different problem (harder, longer chain-of-thought required)
  could shift the threshold higher.
- Decode rate is constant ~101 tok/s — the wall gap is purely
  more reasoning tokens emitted at the same throughput.

Methodology suggestion for the bench harness: a per-case
"effective_budget" probe that samples 2-3 budget values and picks
the smallest one that produces a natural close — typical reasoning
load varies a lot across cases and a one-size budget wastes
compute on easy ones.
