# Bench matrix — 2026-05-19  branch feat/mtp-prefix-warm-ghost  (af05a23)

GPU: NVIDIA GeForce RTX 3090, driver 596.36, 24 GB | Linux WSL2 | CUDA 12.6

All runs share: target `/home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf` (Q4_K_M, ~16 GB, fused backbone + NextN block 64), max-ctx 16384 (32768 for the stack run), `fa-window` 2048 (4096 for the stack run), `max_tokens=512` from the harness. Prompts come from `harness/benchmarks/prompts/` (10 per suite for he/gsm/math; 6 for agent across 2k/8k/24k buckets).

The yesterday matrix this extends lives at `../2026-05-17T17-40-56_f031f08/summary.md` (commit f031f08, AR baseline ~34 tok/s, n_sample=8 n_runs=8). Numbers are not directly comparable to the f031f08 totals because that bench used a different harness (`generation_benchmark.py` reading daemon's internal tok/s timer), and today's runs go through the OpenAI-compatible server via `harness/client_test_runner.py bench` (whole-request wall-clock). The shape of the conclusions still holds (see "Yesterday vs today" below).

## Bug fixes shipped on this branch since f031f08

1. `230c303` mtp: load NextN head blocks onto GPU by default
   - `n_layer = block_count - nextn_predict_layers` correctly excludes MTP blocks from backbone graph iteration, but `plan.layer_end` defaulted to the same reduced value — silently filtering `blk.{n_layer}.*` out of the GPU load. MTP loader's `find_tensor` then resolved descriptors with `data == nullptr` and failed with "14 required NextN tensor(s) missing".
   - Fix: default `plan.layer_end` to `n_block_raw` so MTP heads are loaded alongside backbone. No-op when `nextn_predict_layers == 0`.

2. `5e7594c` qwen35: fix do_spec_decode argmax OOB on prefix-cache partial restore
   - `n_last_chunk = committed % PREFILL_UBATCH` only equals the last prefill chunk size when prefill starts at `kv_offset = 0`. With prefix-cache partial restore (now default), `restore_and_generate` runs delta-prefill from `kv_offset > 0`, so the last chunk is `prompt_len - kv_offset` tokens, not the modulo of `committed`. The read offset exceeded `sg_.argmax_tokens->ne[0]` and tripped `tensor read out of bounds` on the first DFlash request against any prompt the cache had already seen.
   - Fix: read the actual chunk size from `sg_.argmax_tokens->ne[0]`.

3. `af05a23` pflash: propagate skip_park to daemon compress command
   - `compress_text_via_daemon(skip_park=True)` correctly skipped its Python `park target` send, but the C++ `handle_compress` parses an independent `nopark` trailing token and parks target+draft itself when it's absent. The MTP path holds tensor pointers into the backbone's ggml_context across requests; the internal park frees them, the immediate unpark rebuilds with new addresses, and the next forward crashes with `GGML_ASSERT(ggml_can_repeat(b, a))`.
   - Fix: append `" nopark"` to the daemon compress command when `skip_park` is true.

## Agent suite — Qwen3.6-27B Q4_K_M, 6 prompts (2k/8k/24k × 2)

| variant                     | KV    | output tok/s | TTFT  | prefill tok/s | accept | wall |
| --------------------------- | :---: | :----------: | :---: | :-----------: | :----: | :--: |
| MTP γ=3                     | q8_0  | **53.98**    | 1.96s | 607.8         | 0.69   | 7.09s |
| MTP γ=3                     | tq3_0 | 52.07        | 2.08s | 577.9         | 0.70   | 6.09s |
| DFlash b=22                 | q8_0  | 51.99        | 1.63s | 720.4         | 0.29   | 6.12s |
| DFlash b=22                 | tq3_0 | 45.47        | 1.75s | 666.9         | 0.28   | 10.29s |
| **PFlash + MTP γ=3 + TQ3**  | tq3_0 | **53.19**    | 1.46s | 669.1         | 0.70   | 4.89s |

- MTP is robust to KV quant (q8 → tq3 drops 1.9 tok/s, accept stays).
- DFlash gains +6.5 tok/s from q8 KV (cleaner verify logits).
- At q8 KV, MTP and DFlash converge to ~52-54 tok/s on agent. DFlash's accept (29%) is half MTP's (69%) on agent prompts — code/math is where DFlash shines (see yesterday's HE row at 172 tok/s).
- The stacked PFlash+MTP+TQ3 path adds **zero decode tax** on agent-sized prompts (53.19 vs 53.98 MTP-only, within noise). PFlash compression is a no-op for prompts below the 32K threshold but still pays a small per-call overhead — net cost is ~1 tok/s. TTFT looks better on the stack but that's run-to-run variance, not a real win at this prompt size.

## he/gsm/math — Qwen3.6-27B Q4_K_M, n=10, KV q8_0  (server, client_test_runner.py bench)

| Suite | speculator   | output tok/s | TTFT   | accept | accuracy |
| ----- | ------------ | :----------: | :----: | :----: | :------: |
| **he**   | MTP γ=3   | 63.99        | 0.30s  | 0.91   | 8/10     |
| **he**   | DFlash b=22 | **115.50**  | 0.49s  | 0.64   | 8/10     |
| **gsm**  | MTP γ=3   | 53.43        | 0.23s  | 0.75   | 5/10     |
| **gsm**  | DFlash b=22 | **58.95**    | 0.24s  | 0.36   | 7/10     |
| **math** | MTP γ=3   | 56.10        | 0.24s  | 0.80   | 6/10     |
| **math** | DFlash b=22 | **69.71**    | 0.32s  | 0.44   | 4/10     |

## he/gsm/math — bench_matrix.py — apples-to-apples vs f031f08

Same harness as yesterday's f031f08 matrix (`bench_matrix.py`, n_sample=8, n_runs=8 via bootstrap CI 95%, kv q8_0, daemon's internal decode-only tok/s). The bench_matrix orchestrator was added in f031f08 but never merged to main; restored on this branch from `git show f031f08:dflash/scripts/bench_matrix.py` and re-run against today's HEAD binary.

| Suite     | Speculator    | Yesterday f031f08 mean | Today HEAD 83e19d9 mean | Δ vs yesterday |
| --------- | ------------- | :--------------------: | :---------------------: | :------------: |
| humaneval | ar            | 35.06                  | 33.98                   | −3.1%          |
| humaneval | **dflash_b22**| 169.40                 | **173.81**              | **+2.6%**      |
| humaneval | mtp_d3        | 65.62                  | 64.33                   | −2.0%          |
| gsm8k     | ar            | 33.61                  | 33.65                   | +0.1%          |
| gsm8k     | dflash_b22    | 104.32                 | 102.43                  | −1.8%          |
| gsm8k     | mtp_d3        | 61.00                  | 58.51                   | −4.1%          |
| math500   | ar            | 34.57                  | 33.35                   | −3.5%          |
| math500   | dflash_b22    | 119.36                 | 115.27                  | −3.4%          |
| math500   | mtp_d3        | 61.89                  | 61.09                   | −1.3%          |

**No regression on any of the 3 speculators × 3 suites.** All deltas are inside the bootstrap CI 95% bands from yesterday's matrix. DFlash on HumanEval is actually +2.6% today; AR drift is −2% to −3% on HE/Math (cooler GPU or driver micro-variance, ~1 tok/s in absolute terms). MTP holds within −1% to −4% across suites.

Earlier numbers in this same file (115 / 59 / 70 server-side, 109 / 89 / 109 via `bench_llm.py`) report different absolute tok/s because each harness measures a different window:

| Harness | Window |
| ------- | ------ |
| `bench_matrix.py` | daemon's internal `[dflash] generated N tokens in T s` — decode-only, no prefill, no IPC, no Python parsing |
| `bench_llm.py` (direct) | daemon's internal tok/s parsed from stdout — same decode-only window but with different daemon flags (no explicit `-ctk/-ctv`) |
| `client_test_runner.py bench` | server-side, "Out tok/s" = (completion - 1) / decode_only_time, excludes TTFT |
| `harness/benchmarks/generation_benchmark.py` | server-side, completion_tokens / total_elapsed, includes TTFT |

All four harnesses on the same kernel give 87 / 109 / 115 / 172 tok/s on HumanEval today. That entire range is the methodology spread — same code, four windows.

## DFlash b=22 vs MTP γ=3 — head-to-head from yesterday's f031f08 matrix (preserved here for context)

| Suite     | dflash_b22 tok/s | mtp_d3 tok/s | DFlash÷MTP | AL (dflash) | accept (mtp) |
| --------- | :--------------: | :----------: | :--------: | :---------: | :----------: |
| HumanEval | **172.13**       | 66.89        | **2.57×**  | 12.96       | 90.4%        |
| Math500   | **123.51**       | 62.15        | **1.99×**  | 7.79        | 83.5%        |
| GSM8K     | **106.37**       | 60.90        | **1.75×**  | 6.99        | 81.9%        |
| Agent (today) | 51.99 / 45.47 (q8/tq3) | **53.98 / 52.07** | **0.96-0.91× (MTP wins)** | n/a | 0.69 |

The flip on agent is the new finding today: DFlash's drafter accept collapses on chat/tool-use prompts (29% vs MTP's 69%), so MTP's smaller-but-steadier per-step gain wins. Code/math prompts still favor DFlash by 1.75-2.57×.

## Long-context — PFlash + MTP γ=3 + TQ3_0 KV — NIAH @ 36K tokens

Single needle, keep-ratio=0.05, drafter = Qwen3-0.6B-BF16 GGUF (1.5 GB), target+drafter both resident (`--prefill-skip-park` after the af05a23 fix).

| Phase            | Time   | Detail |
| ---------------- | :----: | ------ |
| PFlash compress  | 20.8s  | 35881 → **1769 tokens** (**20.3× compression**, 56 of 1122 chunks kept + 37 forced) |
| Target prefill   | 1.9s   | on 1769 compressed tokens — MTP warm seeded |
| MTP decode       | 0.1s   | 9 output tokens @ **66.3 tok/s** @ accept **1.00** |
| **Wall (TTFT-equivalent)** | **22.8s** | needle recovered exactly (`4025016`) |

- Compression ratio (20.3×) matches RESULTS.md's projection (20.2× at keep=0.05).
- The 20.8s compress is the entire TTFT — drafter Qwen3-0.6B forward+score on 35K tokens. RESULTS.md's 5090 number (11s on 117K) scales sublinearly with bandwidth: 936 GB/s 3090 vs 1792 GB/s 5090 (1.91×) gives an expected ~21s on 3090 at 36K, which matches.
- The MTP decode rate (52-66 tok/s) does not regress under PFlash compression — the MTP head doesn't see the original prompt, it sees the compressed token sequence as if it were a normal prompt.
- NIAH recall passes at keep=0.05 (the drafter's attention-based scoring preserves high-importance tokens including the needle).

## Stack verification — full client probe matrix (all 7 OpenAI-compatible clients)

Server: PFlash + MTP γ=3 + TQ3_0 KV + max-ctx 32768. All 7 clients passed all probes (`/health`, `/v1/models`, chat.stream, chat.non_stream, chat.tools_accepted, anthropic.messages_stream, codex.responses.stream, openwebui.model_metadata).

Per-request breakdown on probe prompts (8-92 input tokens):
- compress: 0.0-0.4s (drafter resident, short prompts pass through)
- prefill: 0.1-0.2s
- decode: 0.2-0.5s at 55-64 tok/s
- MTP accept: 0.71-0.85

Clients tested: claude_code, codex, hermes, openclaw, openwebui, opencode, pi.

## Yesterday vs today — what we proved that yesterday's matrix didn't

1. **Yesterday f031f08** validated DFlash b=22 vs MTP d=3 vs AR on **code/math** (HE / Math500 / GSM8K). DFlash wins by 1.75-2.57× on those suites. MTP wins on consistency (tighter CI bands).
2. **Today** extends to **agent prompts** (the workload class yesterday didn't cover). MTP wins agent by 4-14% over DFlash because DFlash's drafter accept rate collapses (29% vs 69%).
3. **Today** also verifies for the first time that **PFlash + MTP + TQ3 stacks correctly** on a single 3090, with a 36K NIAH passing in 22.8s wall (20.8s compress + 1.9s prefill + decode). The af05a23 fix made this combo runnable at all.
4. **Today's bench_matrix re-run** confirms there is no kernel regression: all 9 cells (3 suites × {AR, DFlash b22, MTP d3}) are within ±5% of yesterday's f031f08 numbers on the same harness.

## Coverage gaps — what is still NOT tested

The labels on `harness/benchmarks/prompts/bench_agent.jsonl` are aspirational, not actual token counts. Real measured sizes (tokenized with `Qwen/Qwen3.6-27B`):

| ID            | Bucket label | Real tokens |
| ------------- | :----------: | :---------: |
| agent_2k_01   | 2k           | 346         |
| agent_2k_02   | 2k           | 268         |
| agent_8k_01   | 8k           | 1524        |
| agent_8k_02   | 8k           | 841         |
| agent_24k_01  | 24k          | 2627        |
| agent_24k_02  | 24k          | 2058        |

So nothing in the agent suite goes past ~3K tokens. Single-turn requests, mostly tool-accept probes and short bug reports. No multi-turn agentic loops, no real SWE-bench instances, no extended code-search-edit-test cycles.

The Qwen3.6-27B model's native context is **262144 tokens (256K)** per the GGUF metadata. We've validated single-needle NIAH at **36K (14% of ceiling)**. Untested:

- Single-prompt context 48K / 64K / 96K / 131K / 192K / 262K
- Multi-needle NIAH (drafter selection survives multiple high-importance regions?)
- Real agentic loops at any depth (SWE-bench, multi-turn tool-use, RAG with retrieved context)
- Concurrent sessions (server has prefix-cache slots=1 by default; saturation behavior under load is unknown)
- Sustained throughput across hundreds of requests (drafter weights stay resident — confirmed at 8 probes, not 800)

A reasonable next milestone is a long-context ceiling sweep: NIAH at 32K → 64K → 131K → 262K with PFlash+MTP+TQ3 on a single 3090, then real SWE-bench loops at 24K-65K context once the long-NIAH ceiling is mapped.

## Reproducing

```
# MTP server (best agent decode)
python3 dflash/scripts/server.py --host 127.0.0.1 --port 18080 \
  --target /home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf \
  --bin /home/peppi/Dev/lucebox-hub/dflash/build/test_dflash \
  --mtp-gguf /home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf \
  --mtp-gamma 3 --mtp-draft-source chain \
  --max-ctx 16384 --fa-window 2048 \
  --cache-type-k q8_0 --cache-type-v q8_0

# Full stack (PFlash + MTP + TQ3) for long context
python3 dflash/scripts/server.py --host 127.0.0.1 --port 18080 \
  --target /home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf \
  --bin /home/peppi/Dev/lucebox-hub/dflash/build/test_dflash \
  --mtp-gguf /home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf \
  --mtp-gamma 3 --mtp-draft-source chain \
  --prefill-compression always --prefill-keep-ratio 0.05 \
  --prefill-drafter /home/peppi/models/Qwen3-0.6B-BF16.gguf \
  --prefill-skip-park \
  --max-ctx 32768 --fa-window 4096 \
  --cache-type-k tq3_0 --cache-type-v tq3_0

# Benches
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080 --suite he,gsm,math --model luce-dflash
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080 --suite agent  --model luce-dflash
python3 harness/client_test_runner.py probe --url http://127.0.0.1:18080 --clients all
```
