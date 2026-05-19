# Dual-Speculator Architecture & Open Questions — 2026-05-19

Working doc for the MTP + DFlash + PFlash + TQ3 stack on Qwen3.6-27B,
priority hardware RTX 3090 24 GB, priority workload agentic coding via
hermes / opencode / pi CLIs.

## 1. The full pipeline (per request)

```
USER CLI (hermes / opencode / pi / claude / codex)
   │ POST /v1/chat/completions  { messages, max_tokens, speculator? }
   ▼
dflash/scripts/server.py  (FastAPI + SSE, OpenAI + Anthropic compat)
   • reads body.speculator (or extra_body.speculator)
   • prefix-cache lookup per conversation
   ▼ stdin protocol
test_dflash daemon  (Qwen35Backend, single CUDA context)
   │
   ├── PFlash COMPRESS  ← only when prompt > 32K (or `always`)
   │   Qwen3-0.6B-BF16 scorer, BSA attention
   │   keep_ratio 0.05 → 20× compression typical
   │   20-100 s TTFT cost on 3090 depending on input size
   │
   ▼
   TARGET PREFILL
   Qwen3.6-27B-Q4_K_M (15.3 GB)
   KV cache: TQ3_0 / Q8_0 / Q4_0
   FA window 4096, chunked ubatch 512
   prefix-cache restore if WARM hit
   │
   ▼
   auto_select(prompt_len > 4096 ?)
        │                         │
        ▼ ≤4096                   ▼ >4096
   DFlash drafter            MTP NextN heads
   1 GB, AL 7-13 on code     0.5 GB, accept 0.7-1
                             prompt-agnostic
        │                         │
        └──────────┬──────────────┘
                   ▼
   TOKEN-BY-TOKEN VERIFY (target forward on γ+1 / DDTree)
   argmax → accept-run → sample sentinel
   │
   ▼ emit accepted tokens via stream_fd
   SSE chunks → CLI
```

## 2. Component cost (RTX 3090 24 GB)

| Component                       | VRAM     | Latency tax | Wins on            | Hurts on              |
|---------------------------------|----------|-------------|--------------------|-----------------------|
| Target Qwen3.6-27B Q4_K_M       | 15.3 GB  | baseline    | all                | —                     |
| KV TQ3_0 (16K ctx)              | ~1.5 GB  | −3 %        | long ctx           | —                     |
| KV Q8_0 (16K ctx)               | ~3.0 GB  | 0           | short / code       | —                     |
| KV Q4_0 (16K ctx)               | ~1.0 GB  | ~0          | extreme ctx        | —                     |
| DFlash drafter Q4 GGUF          | 1.0 GB   | +3-5 ms/step| code, math         | chat, compressed ctx  |
| MTP NextN heads (in target)     | 0.5 GB   | +2-3 ms/step| chat, long ctx     | — (cheap default)     |
| PFlash compress (0.6B BF16)     | +1.5 GB  | +5-100 s    | ctx > 32K          | short ctx (no gain)   |
| Prefix cache (per slot)         | ~30 MB   | saves prefill| multi-turn        | —                     |

Dual-spec + PFlash + TQ3 KV total: 20.3 GB → fits 24 GB.

## 3. Workload → recommended config

```
prompt < 4K AND structured (code / math) ───► DFlash b=22 (115-180 tok/s)
prompt < 4K AND chatty / tool-use ─────────► MTP γ=3      (50-65  tok/s)
4K ≤ prompt < 32K (accumulated history) ──► MTP γ=3      (50-65  tok/s)
prompt ≥ 32K (RAG / long code review) ────► PFlash + MTP (43-66  tok/s)
                                              + 20-100 s TTFT
                                              + 20× compression
```

Backend `auto` mode threshold: `prompt_tokens > 4096 → MTP`, else DFlash.

## 4. Synergies vs conflicts

|                  | DFlash | MTP    | PFlash      | TQ3    | Prefix$    | FA win |
|------------------|--------|--------|-------------|--------|------------|--------|
| **DFlash**       |   —    | ✅ both| ⚠ accept    | ✅     | ✅         | ✅     |
|                  |        | load OK| −50 % on    |        |            |        |
|                  |        |        | compressed  |        |            |        |
| **MTP**          | ✅     |   —    | ✅ no drop  | ✅     | ✅ (WARM)  | ✅     |
| **PFlash**       | ⚠ avoid| ✅ best |  —          | ✅     | ✅         | ✅     |
| **TQ3 KV**       | ✅     | ✅     | ✅          |   —    | ✅         | ✅     |
| **Prefix cache** | ✅     | ✅     | ✅          | ✅     |   —        | ✅     |
| **FA window**    | ✅     | ✅     | ✅          | ✅     | ✅         |   —    |

**The one real conflict** is DFlash × PFlash: drafter accept rate drops
from 0.91 (uncompressed code) to 0.14-0.21 (PFlash-compressed long ctx).
auto-select avoids it by routing prompt > 4K to MTP. But that's
avoidance, not a fix.

## 5. The DFlash × PFlash conflict — why it happens

PFlash compress keeps the top-k token POSITIONS by attention-importance.
The kept sequence is a sparse, gapped subset of the original tokens:

```
Original:    [<bos>, "def ", " compute", "(", "x", ",", " y", "):"]
Compressed:  [<bos>, "def ", "compute", ":"]   ← gaps preserved as continuous
```

DFlash drafter is a Block-Diffusion next-token-prediction model trained
on **natural sequences**. Given a compressed prompt, its local context is
mangled — next-token entropy is high, accept rate collapses to 14-21 %.

MTP heads don't see raw tokens — they consume the **backbone's hidden
states**. After the backbone has processed the compressed prompt, the
hidden states already encode "this is a compressed input." MTP heads
read those rich h-states and predict normally. The compression
distortion is largely absorbed by the backbone, not the speculator.

## 6. Solving the conflict — three paths

### Path A — Avoidance via auto_select (shipped)

`auto_select(prompt_len > 4096) → MTP`. DFlash never sees a compressed
prompt because PFlash only fires above 32K (or `always` mode) and
auto-select hands off to MTP at 4K. **Effective, simple, no extra cost.**

### Path B — Re-train DFlash drafter on compressed sequences

Generate a corpus of (uncompressed prompt, PFlash-compressed prompt)
pairs, fine-tune the dflash-draft-3.6 drafter on the compressed
sequences with teacher-forcing on target's hidden states. Probably
recovers most of the accept rate, but requires training infrastructure
and the dflash drafter is closed-format (z-lab safetensors).

Effort: HIGH.

### Path C — Replace PFlash drafter with a DIFFERENT scorer

The PFlash drafter only needs to produce per-token importance scores
via BSA attention. It does NOT need to predict next tokens. We can swap
it for any model whose attention pattern aligns better with the target's.

**Candidates already supported by the loader** (see
`dflash/src/qwen3/qwen3_drafter.cpp:parse_drafter_arch`):

| arch        | model                                | size  | rationale                                |
|-------------|--------------------------------------|-------|------------------------------------------|
| qwen3-0.6b  | Qwen3-0.6B-BF16.gguf (current)       | 1.5 GB| baseline                                 |
| qwen35-0.8b | Qwen3.5-0.8B-Q4_K_M.gguf (on disk)   | 508 MB| smaller, same tokenizer family           |

**Candidates that would need a new loader path:**

- z-lab dflash-draft-3.6 (1 GB, already loaded for DFlash decode) — would
  let one model do BOTH compress AND decode drafting. Block-Diffusion
  architecture; needs a BSA path. **HIGH-VALUE.**
- Qwen3.6-27B-MTP heads as a scorer — but they consume hidden states, not
  raw tokens, so they'd need a separate cheap-forward path (e.g. only the
  first 2-3 backbone layers) to produce h-states cheaply.
- Smaller fine-tune of the target model itself (~3B) trained specifically
  on importance scoring.

### Path D — Same model for both PFlash scoring AND DFlash decode drafting

The clean win. Currently:
- PFlash drafter:  Qwen3-0.6B-BF16  (1.5 GB)  — attention-based scoring
- DFlash drafter:  dflash-draft-3.6 (1.0 GB)  — next-token prediction

If we could use ONE drafter for both jobs:
- VRAM saved: 1.5 GB (drop the Qwen3-0.6B)
- Tokenizer perfectly aligned (same vocab, same encoding)
- The drafter's compress-time KV cache could prime decode-time KV (cache reuse!)
- One model to validate, one place to upgrade

**Requirements for the unified drafter:**
1. Same tokenizer as target ✓ (dflash drafter shares Qwen3.6 vocab)
2. Block-Sparse Attention path for compress (needs implementation if dflash drafter doesn't have it)
3. Long-context capability for compress: must handle 32K-131K inputs.
   Current dflash drafter context: ~4K with SWA. **Limiting.**
4. Decent next-token-prediction in target's space ✓ (that's what dflash drafter does)
5. Sufficient parameter count to produce useful importance scores (1B params
   is borderline; Qwen3-0.6B works because the BSA pattern matters more than
   the model size)

**Likely best concrete experiment to run first:**

| Experiment | Setup | Expected signal |
|------------|-------|-----------------|
| E1: Qwen3.5-0.8B as PFlash drafter | swap `--prefill-drafter` to `/home/peppi/models/qwen3.5-0.8b-draft/Qwen3.5-0.8B-Q4_K_M.gguf` + `--drafter-arch qwen35-0.8b` | does NIAH recall hold? compress time? compression ratio? |
| E2: dflash drafter as PFlash drafter | add a Block-Diffusion BSA path; swap | does Path D work? same-tokenizer win? |
| E3: cache reuse (PFlash drafter's KV → DFlash drafter's KV)| both jobs use the same drafter, persist KV across compress→decode | TTFT win on multi-turn with same prefix |

E1 is the cheapest first cut. If Qwen3.5-0.8B works as well as Qwen3-0.6B
(and it's a third the size), we save 1 GB and validate that PFlash scoring
is relatively model-agnostic. That alone would shift the cost matrix
significantly.

## 7. What the user sees (recommended 3090 default for agentic coding)

```
python3 dflash/scripts/server.py --host 127.0.0.1 --port 18080        \
  --target  $TARGET  --mtp-gguf $TARGET  --mtp-gamma 3                \
  --draft   $DFLASH_DRAFT_GGUF  --budget 22 --verify-mode ddtree      \
  --bin     dflash/build/test_dflash                                  \
  --prefill-compression auto                                          \
  --prefill-threshold 32000  --prefill-keep-ratio 0.05                \
  --prefill-drafter $PFLASH_DRAFT_BF16  --prefill-skip-park           \
  --cache-type-k q8_0  --cache-type-v q8_0                            \
  --max-ctx 32768  --fa-window 2048                                   \
  --prefix-cache-slots 4
```

Across a real 25-turn hermes/opencode/pi session:

```
Turn 1   user prompt 800 tok   ── auto: DFlash b=22  ──► ~100 tok/s
Turn 2   prompt 1.6K (history) ── DFlash             ──► ~80  tok/s
Turn 8   prompt 4.2K           ── crosses threshold  ──► MTP @ ~55 tok/s
Turn 12  prompt 9K             ── MTP                ──► ~50 tok/s
Turn 25  prompt 35K            ── PFlash + MTP       ──► 20 s compress + 50 tok/s
```

## 8. Known gaps / what we DON'T solve

- **Drafter resident under 16 GB VRAM**: dual-spec fits 24 GB but not 16
  (would need to park one speculator). Need a `--memory-tight` profile.
- **Auto threshold is naïve** (just prompt_tokens > 4K). A smarter
  selector could read system-prompt fingerprint or per-CLI accept history.
- **Mid-stream switch**: if DFlash starts low-accept on a single long
  generation, we don't switch to MTP mid-stream. Option B (live accept
  feedback → fallback) is the next implementation step.
- **No cross-session telemetry**: server doesn't learn that "this user's
  hermes-coding loop always benefits from MTP." Every session starts cold
  on the auto heuristic.
- **PR #195** (draft safetensors loader Qwen3.6 fix) not merged — our
  GGUF path is safe, safetensors crashes on Qwen3.6.
- **No real CLI session tested yet**. Driver at
  `/tmp/bench-runs/cli_session_driver.py` is ready; need server up and
  user willing to drive.

## 9. Next concrete experiments (in cost order)

| # | Experiment                                                  | Time est | Decision it answers |
|---|-------------------------------------------------------------|----------|---------------------|
| 1 | Swap PFlash drafter to Qwen3.5-0.8B-Q4 (508 MB); rerun NIAH 36K | ~10 min | Is Qwen3-0.6B over-spec'd for the job? |
| 2 | Drive 1 CLI (start with `pi`, smallest output) on dual server | ~20 min | Does real tool-use loop break anything? |
| 3 | Add per-CLI driver script results to bench dir              | ~10 min | Baseline real-usage numbers |
| 4 | Add a Block-Sparse Attention path to dflash drafter         | ~3 hr    | Path D feasibility |
| 5 | Bench Path D (one drafter for both jobs) on NIAH 36K+131K   | ~30 min  | VRAM win + accept compatibility |
| 6 | Implement Option B (mid-stream fallback on low accept)      | ~1 day   | More robust auto-select |
| 7 | Open PR; merge PR #195 as dep                               | ~30 min  | Ship to main |
