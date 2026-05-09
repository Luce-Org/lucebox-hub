# Gemma4 context-scaling test plan: 1k → 4k → 8k → 32k → 256k

**Status going in:** Bug 1 (SWA mask for n_tokens==1 decode, parent `7b62c07`) + Bug 2 (TQ3 K dispatcher → MMA intercept FWHT mismatch, submodule `d758ed9bf`) are fixed and verified at small context. **MTP+TQ3/TQ3 now works**: accept_rate 0.56 on 64 decode steps, coherent prose. Target+TQ3/TQ3 also coherent. Q8/Q8 unchanged.

**Goal:** map performance + correctness across context lengths to find where the new fixes hold, where they break, and the highest-context production-ship config for the MoE + pFlash demo.

**Hardware:** RTX 3090 24GB. CUDA 13.1, sm_86.

---

## Phase 0 — preflight

Before any runs:

1. **Build is clean at HEAD `7b62c07`**: rebuilt during this session, binary at `dflash/build/test_gemma4_dflash`.
2. **Real tokenized prompts**: `short_chat.txt` (27 tok), `long_open.txt` (40 tok), `long_2k.txt` (2,611 tok) under `.sisyphus/notes/gemma4-baseline/prompts/`. **More prompts needed for ≥4k tests** (see Phase 4).
3. **Models available**:
   - 31B dense target: `models/gemma-4-31B-it-Q4_K_M.gguf` (~18 GB)
   - 31B MTP drafter: `models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q8_0.gguf`
   - **31B dflash drafter**: `/home/peppi/models/draft-gemma4-31b/` — **NOT TESTED YET this session**
   - MoE target: `/home/peppi/models/gemma4-26b-a4b-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` (~13 GB)
   - MoE dflash drafter: `/home/peppi/models/gemma4-26b-a4b-dflash/draft-q8_0.gguf`
   - **MoE has NO MTP drafter** — MoE will run with dflash only (or no drafter).
4. **Deferred bugs not blocking this plan but to flag during runs**:
   - FA-kernel-selection abort at `fattn.cu:652` for head_dim=512 + Q8 KV + non-aligned KV-length. Hit M4 at step ~210, will likely hit any 31B-dense + MTP + Q8/Q8 run that crosses similar context. Workaround: use TQ3 KV (now fixed) or stay below the trigger context.

---

## Phase 1 — short context: 1k, 4k, 8k

These all fit comfortably in VRAM regardless of KV type. Goal: lock in the baseline numbers across configs, confirm fixes hold past short tests, find the FA-crash trigger context for Q8/Q8.

### Configs (each at all 3 context lengths)

| Cell | Drafter | KV-K | KV-V | Why we run it |
|------|---------|------|------|---------------|
| A | none | Q8 | Q8 | Target-only baseline (no draft overhead) |
| B | none | TQ3 | TQ3 | Target-only with full TQ3 (memory savings) |
| C | mtp  | Q8 | Q8 | Best-known MTP accept_rate (was 0.65-0.68 at small ctx) |
| D | mtp  | TQ3 | TQ3 | **Headline production target** (was 0.56 at 64 steps) |
| E | dflash | Q8 | Q8 | The leg we missed — dflash on dense 31B |
| F | dflash | TQ3 | TQ3 | dflash + TQ3 — interesting if it inherits the fixes too |

### Context sizes + prompt sizing

| Ctx target | --ctx-size | Prompt tokens | n_predict | Notes |
|------------|------------|---------------|-----------|-------|
| 1k         | 4096 (default) | 41 (`long_open.txt`) | 256 | Already covered by matrix-v3 — re-baseline only if needed |
| 4k         | 4096 | 2611 (`long_2k.txt`) | 256 | Stresses prefill + decode at ~3k effective ctx |
| 8k         | 8192 | ~6500 (need new prompt: `long_8k.txt`) | 512 | Need new prompt |

### Pass criteria

- Cell completes without crash at each context size.
- Decoded output (first 80 tokens) is coherent English (manual check).
- For drafter cells: accept_rate stable across 32-step windows (no slow collapse, no late spike from looping).
- Final tok/s recorded.

### Numbers to record per cell

`prefill_ms`, `prefill_tok/s`, `decode_ms`, `decode_tok/s`, `first_tok_ms`, `VRAM_used_GB`, `accept_rate_final` (drafters only), `coherent_yes_no`.

---

## Phase 2 — medium context: 32k

VRAM starts mattering: at 32k, Q8 KV ≈ 2 GB, TQ3 KV ≈ 0.9 GB. Both fit.

### What we expect to break

- **Cell C (MTP+Q8/Q8) at 32k**: very likely hits the FA-kernel-selection abort. The crash trigger is head_dim=512 + Q8 KV + non-FATTN_KQ_STRIDE-aligned KV-length. At 32k, alignment reaches non-aligned values frequently. Document the crash step.
- **Cells E/F (dflash drafter)**: unknown territory. Worth testing.

### Prompts needed

- `long_30k.txt` (~30k tokens) — generate from a Project Gutenberg novel chapter or similar reproducible source.

### Configs to run

Same A-F as Phase 1, with `--ctx-size 32768`, prompt = `long_30k.txt`, `n_predict 1024`.

### Bonus: --pflash flag

The driver supports `--pflash` for prompts ≥4096 tokens. At 32k, this is the natural domain. Add a `pflash` variant of cells A and C: `A_pflash` and `C_pflash`. Compare prefill tok/s with vs without pflash.

---

## Phase 3 — long context: 256k

This is where memory tightens hard:

| KV type | Per-token KV bytes (60 layers × 2 × 256 × 2) | At 256k tokens | Plus 18 GB target | Fits 24GB? |
|---------|--------|------|------|-------|
| F16     | 122,880 | ~32 GB | ~50 GB | ❌ |
| Q8      | 61,440  | ~16 GB | ~34 GB | ❌ |
| **TQ3** | ~26,880 | **~7 GB** | **~25 GB** | tight, possibly ✓ |

(Numbers approximate; head_dim and n_head_kv vary per layer.)

**For 31B dense at 256k**: only TQ3 KV fits. Run cells B (target-only TQ3) and D (MTP+TQ3) ONLY. Skip Q8 cells.

**For MoE 26B-A4B at 256k**: smaller weights (~13 GB) leave more headroom. Q8 KV plausibly fits. Test both.

### Prompts needed

- `long_200k.txt` (~200k tokens) — concat several Project Gutenberg books, run through HF tokenizer.

### Configs

| Cell | Model | Drafter | KV-K | KV-V | --pflash | n_predict |
|------|-------|---------|------|------|----------|-----------|
| G    | 31B dense | none | TQ3 | TQ3 | yes | 256 |
| H    | 31B dense | mtp  | TQ3 | TQ3 | yes | 256 |
| I    | MoE 26B   | none | TQ3 | TQ3 | yes | 256 |
| J    | MoE 26B   | dflash | TQ3 | TQ3 | yes | 256 |
| K    | MoE 26B   | dflash | Q8 | Q8 | yes | 256 | (if VRAM fits) |

### Pass criteria

- Each cell completes prefill (the long prefill is the hard part).
- Decode produces coherent output (first 80 tokens of generation).
- Prefill tok/s recorded; pFlash speedup quantified.
- `--pflash-alpha` left at default 0.12 unless we want to sweep.

---

## Phase 4 — prompt manufacturing (do this BEFORE Phase 1c+)

The benchmark depends on real BPE-tokenized long prompts. Generate them once, reuse:

| File | Source | Target tok count |
|------|--------|--------|
| `long_8k.txt`   | Alice in Wonderland Chs 1-3 | ~6500 |
| `long_30k.txt`  | Alice in Wonderland full + Hunting of the Snark + a few short stories | ~30000 |
| `long_200k.txt` | Multiple Gutenberg novels concatenated under one chat-template wrapper | ~200000 |

Use the existing `generate_prompts.py` under `.sisyphus/notes/gemma4-baseline/` as the template (HF `google/gemma-3-27b-it` tokenizer; chat-template wrap; CSV output). Sidecars `.meta` for each.

**VRAM cap**: even with TQ3, a 200k-token prompt + 256k ctx alloc is ~25 GB. The prompt may need to be capped at e.g. 180k to leave headroom for generation.

---

## Phase 5 — recovery scripts

Write small shell scripts under `.sisyphus/notes/gemma4-baseline/`:
- `run_phase1.sh` — runs cells A-F at 1k, 4k, 8k. Captures all stats. Builds a side-by-side report.
- `run_phase2.sh` — same for 32k, plus `--pflash` variants of A and C.
- `run_phase3.sh` — 256k cells G-K.

Each script writes results to `matrix-v4/<cell>_<ctx>k.log` and a single `SUMMARY.md` table. Use `--temp 0 --seed 0 --ignore-eos` everywhere for reproducibility.

---

## Risks + mitigations

1. **FA-kernel-selection abort** for Q8/Q8 + head_dim=512 + non-aligned KV (the M4 crash) — likely hits Phase 2 cells C/E and Phase 3 K. Mitigation: keep the run going on TQ3 cells; document the abort step for each.
2. **256k prompt won't tokenize cleanly** — chat template + extreme length might exceed model's training distribution; output may be incoherent regardless of correctness fixes. Mitigation: report what we see; the *prefill timing* is still the headline number even if generation is off-topic.
3. **TQ3 chunked path is slower than MMA** — Phase 1 Q8 vs TQ3 throughput gap (~2× decode hit) will compound at long context. The win is *that it runs at all* in 24 GB; raw speed is secondary at 256k.
4. **MoE branch routing** — never validated end-to-end this session. Phase 3 MoE cells may surface fresh bugs unrelated to KV.

---

## Decision gates

- After Phase 1: if any cell A-F regresses or crashes at small context, debug before scaling up.
- After Phase 2: if FA crash at 32k blocks Q8/Q8 MTP, defer Q8 from Phase 3. Continue with TQ3 only.
- After Phase 3: report findings; the highest-context working ship config becomes the demo target.

---

## Quick-start commands (for next session)

```bash
# Phase 1, cell D, 1k ctx (the headline regression test)
./dflash/build/test_gemma4_dflash \
  --model models/gemma-4-31B-it-Q4_K_M.gguf \
  --mtp models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q8_0.gguf \
  --draft-method mtp --kv-k tq3_0 --kv-v tq3_0 \
  --tokens-file .sisyphus/notes/gemma4-baseline/prompts/long_open.txt \
  --n-predict 256 --temp 0 --seed 0 --ignore-eos

# Phase 1, cell E (the missed leg — dflash drafter on dense 31B)
./dflash/build/test_gemma4_dflash \
  --model models/gemma-4-31B-it-Q4_K_M.gguf \
  --draft /home/peppi/models/draft-gemma4-31b \
  --draft-method dflash --kv-k q8_0 --kv-v q8_0 \
  --tokens-file .sisyphus/notes/gemma4-baseline/prompts/long_open.txt \
  --n-predict 256 --temp 0 --seed 0 --ignore-eos

# Phase 3, cell I (MoE target-only at 256k — needs long_200k.txt)
./dflash/build/test_gemma4_dflash \
  --model /home/peppi/models/gemma4-26b-a4b-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --draft-method none --kv-k tq3_0 --kv-v tq3_0 \
  --tokens-file .sisyphus/notes/gemma4-baseline/prompts/long_200k.txt \
  --ctx-size 262144 --pflash --n-predict 256 --temp 0 --seed 0 --ignore-eos
```
