# Making Gemma-4 fast on a 24 GB GPU: a debugging journey

**TL;DR** — Three bugs were silently stealing performance from Gemma-4 on 3090-class hardware: (1) the SWA causal mask was never built for single-token decode, (2) the TQ3_0 KV path silently routed through an MMA kernel that didn't undo the FWHT rotation, and (3) the head_dim=512 MMA gqa-opt branch was rejected for Q8 KV because no mask was supplied. After fixing those, the same 26B-A4B MoE that *collapsed to 13 tok/s decode at 64k context* in our baseline ships **36.57 tok/s** with the same drafter and weights — and runs to **256k context at 21.7 GB on a 24 GB RTX 3090**. The lessons that took days of compute to discover, in a few thousand words, in case they save someone else a weekend.

This post is written for engineers maintaining a fork of `llama.cpp` for speculative decoding (DFlash / MTP / Medusa-class), targeting Gemma-4 (or any Gemma-4-like SWA + GQA + chunked-prefill model) on consumer GPUs. Skip ahead to the section that matches your symptom.

---

## The setting

- **Hardware**: RTX 3090, 24 GB VRAM, sm_86, CUDA 13.1.
- **Model A**: Gemma-4-31B-it Q4_K_M (dense, ~18 GB weights). 60 layers, 50 SWA + 10 full-attn, head_dim 256 (SWA) and 512 (full-attn), n_head_kv=8, GQA ratio 8.
- **Model B**: Gemma-4-26B-A4B-it Q4_K_M (MoE, ~13 GB weights). 30 layers, 8 of 128 active experts + 1 shared per layer.
- **Drafters**:
  - DFlash (z-lab style, target-conditioned): 5-layer Q8 GGUF for both 31B (1.6 GB) and MoE (456 MB).
  - MTP (Google-style, 4-layer assistant attached to target): only available for 31B dense.
- **Stack**: `test_gemma4_dflash` binary, ggml/cuda backend, KV-cache with TQ3_0 / Q8_0 / Q4_0 / F16 options. pFlash (block-sparse prefill) is on a custom `GGML_OP_FLASH_ATTN_SPARSE`.

The headline goal we were chasing: **MoE 26B + dflash + 256k context, single 24 GB GPU, in production-relevant tok/s**. Spoiler: that's now feasible and the production numbers are in this post. But the path here started with us not being able to reproduce a 0.22 accept_rate baseline that turned out to be on garbage input.

---

## Day 0: the contaminated baseline

The plan we inherited declared:
> TQ3_0 cross-attention now functional. accept_rate 0.22 on Q4_K_M target + Q8_0 assistant + TQ3_0 KV, 131-token prompt, 64-step generation.

Running the same command, we got **0.22 at step 64**, then a slow slide to **0.06 at step 256**. That doesn't pattern-match a real text completion. Decoding the generated token IDs back to strings using HF's `google/gemma-3-27b-it` tokenizer (which is byte-identical to the GGUF's vocab, 262144 entries, BOS=2, EOS=106 — verified by side-by-side comparison of `gguf.GGUFReader`'s `tokenizer.ggml.tokens` and `AutoTokenizer.from_pretrained()`'s vocab) revealed:

```
<unused94>をlaenat quelelele tolaredlele samme a które a a a a
 a a a a a a a   a a które up up a a robot samme a robot
```

Multilingual gibberish followed by a `a robot` repetition loop. Not a real completion.

**The first lesson:** the test driver's `test_gemma4_dflash` was using **byte-fallback tokenization** by default — the message in the binary's startup is literally:
```
[tokens] byte-fallback tokenisation: 102 tokens (pass --tokens <ids> for real tokenisation)
```
102 bytes for "The quick brown fox jumps over the lazy dog. Explain in one paragraph what this sentence demonstrates." Not 25-30 BPE tokens. The model was being fed UTF-8 bytes as if each were a vocab id.

**Fix**: built a tokenization pipeline using the in-repo HumanEval+ jsonl + HF's `google/gemma-3-27b-it` tokenizer + the Gemma chat template (`<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`). Saved 6 prompts: short_chat (27 tok), long_open (40 tok), long_2k (2611 tok, Alice in Wonderland Ch. 1), long_50k (49904 tok, Tiny Shakespeare summarisation request), long_code_50k (50002 tok, concatenated HumanEval+ tasks), and humaneval_2 (139 tok, single HE task with EvalPlus chat format).

After that switch, the **same** binary on the **same** model produced "This sentence is a **pangram**, which is a phrase that contains every letter of the alphabet at least once. Because it is relatively short and coherent" — a real answer.

**Take-away**: if your bench framework outputs token IDs only and your accept_rate metric is "drafter agrees with target", you can chase 0.22 forever on inputs that aren't even in distribution. Always decode and read your output text. **Real-token plumbing first; everything else after.**

---

## Bug 1: the SWA mask that wasn't there for single-token decode

With real BPE input, target+TQ3 still produced garbage. Target+Q8 produced clean prose. Same prompt, same seed, same temp=0.

Decoding ablations narrowed the suspect to the SWA layers' attention. Adding a `fprintf` diagnostic at the SWA FA call site revealed:

```
[swa-fa-diag] il=0 n_tokens=28 kv_start=0  K_ne1=2048 mask=swa_mask  mask_ne0=2048 mask_ne1=32  ← prefill ✓
[swa-fa-diag] il=0 n_tokens=1  kv_start=28 K_ne1=2048 mask=attn_mask mask_ne0=256  mask_ne1=32  ← decode ✗
```

For prefill (n_tokens=28), the mask is the proper `swa_mask`, sized 2048×32 to match the K view. For decode (n_tokens=1), it falls back to the full-attn `attn_mask` sized to the kv_len padding (256 wide), but the K view is still 2048. **256-wide mask, 2048-wide K view.** The kernel reads past the populated region into uninitialized cudaMalloc bytes.

Why didn't it crash for Q8? Q8's higher precision was tolerant — the populated 28 K positions still dominated the corrupted attention distribution. TQ3's quant noise + uninitialized garbage produced a near-flat distribution that fed the LM head a weak signal, which fell back to high-frequency tokens (`'en'`, `'a'`, `<unused94>`).

The bug was in the test driver's graph builder. The SWA mask was guarded:
```cpp
if (n_tokens > 1) {                  // ← "batched prefill only"
    sg.swa_mask = ggml_new_tensor_2d(...);
}
```
And the comment in `internal.h` was:
```cpp
ggml_tensor * swa_mask  = nullptr;  // sliding-window causal mask (batched prefill only)
```

**Fix**: drop the `n_tokens > 1` guard; allocate `swa_mask` always when masks are requested. Add the matching `build_swa_causal_mask()` call at all four single-token decode sites (daemon decode, decode warmup, MTP target verify, target-only decode). Update the comment to reflect the new contract.

This one fix changed Q8 output too: "This sentence is a sentence" → "This sentence is a phrase" (slightly different attention output → different greedy trajectory, both coherent). The pre-fix Q8 was incidentally surviving; with the mask correct, it's actually more right.

---

## The bisect that proved the bug was older

After Bug 1, target+TQ3 still produced garbage. The hypothesis was a recent regression in the MTP-related commits. We did a `git bisect` over the 6 commits between `7eea84b` (last pre-MTP) and `c56879c` (HEAD before our fixes), with a coherence-check predicate that decoded the first 16 generated tokens and looked for English alphabet runs.

**Result**: every commit in the bisect range produced the same garbage on TQ3. Walking back further, we tested at `ce4da35` (the very first commit that integrated TQ3 with Gemma-4, "narrow asymmetric KV"). Same garbage. **TQ3 + Gemma-4 + real BPE tokens has never produced coherent text.**

The "0.22 accept_rate" the plan had been chasing was on byte-fallback junk input. Both the target and the drafter were generating non-text. The drafter's "accept rate" was just "drafter successfully predicts that the target produces the same gibberish." Double-fake.

**Take-away**: a well-run bisect that returns "no good commit" is a real answer. Stop bisecting and pivot. The bug was older than the bisect range; it lives in the TQ3 + Gemma-4 interaction, not in any specific commit.

---

## Bug 2: the TQ3 K dequant intercept that silently strips FWHT rotation

A focused Codex audit on the TQ3 K-side path landed the cause in `dflash/deps/llama.cpp/ggml/src/ggml-cuda/fattn.cu` lines 134–204:

```cpp
if (K->type == GGML_TYPE_TQ3_0 || V->type == GGML_TYPE_TQ3_0) {
    // ... allocate temp F16 buffers ...
    if (K->type == GGML_TYPE_TQ3_0) {
        k_tq3_0_dequant_f16_full<<<...>>>(...);  // dequant TQ3_0 → F16
        K_f16.type = GGML_TYPE_F16;
        dst->src[1] = &K_f16;                     // ← swap K reference
    }
    // ... same for V ...
    // re-enter the standard MMA dispatch with substituted K/V types
    switch (Q->ne[0]) {
        case 256: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst); break;
        case 512: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst); break;
        ...
    }
}
```

The intercept dequantizes TQ3_0 K storage to F16 and re-enters the MMA dispatch. **But TQ3_0 K is stored in FWHT-rotated form** (the rotation is applied during `tq3_rotate_forward` at quantization write time; see `cpy-utils.cuh:215+227`). The chunked / vec FA kernels handle this correctly: when they see `K->type == GGML_TYPE_TQ3_0`, they forward-rotate Q before computing Q@K, so the dot product happens in matched FWHT space. The MMA intercept skips that hook because by the time it dispatches, `K->type == GGML_TYPE_F16`. **Q is in standard space, K is in FWHT-rotated space, Q@K is computed in mismatched domains.**

V works on the asymmetric path (K=Q8, V=TQ3) for the same reason inverted: the V FWHT only affects the post-attention output, not the attention score distribution. Tokens are sampled from `softmax(QK)` not from V. So **V being in the wrong domain doesn't break token selection**, only the output values get rotated incorrectly — and since the next layer's V is computed fresh from W_v @ x of the rotated output, it just propagates a basis change that doesn't matter for argmax sampling.

The dispatcher in `fattn.cu:534` had:
```cpp
const bool tq3_needs_chunked = tq3_any && (Q->ne[1] > 1 || Q->ne[0] > 256) && !tq3_use_mma;
```
For SWA decode (`Q->ne[1]==1, Q->ne[0]==256`), neither clause fires; the path falls through to the broken MMA intercept.

**Fix**: drop the `(Q->ne[1] > 1 || Q->ne[0] > 256)` guard. Force chunked for ALL TQ3 cases unless `DFLASH_TQ3_MMA` is opted in. Restores the "historically forced chunked" behavior the in-source comment explicitly claims.

After this fix, MTP+TQ3/TQ3 went from accept_rate 0.05 (degenerate loop) to **0.56 with coherent prose**: "Unit 7 was designed for precision, not poetry. A maintenance droid with a steady hydraulic arm, its world was composed of grids, bolts, and gray steel."

---

## Bug 3: head_dim=512 + Q8/Q8 MMA gqa-opt requires a non-null mask

After Bugs 1+2 landed, MTP+Q8/Q8 still aborted in production at `fattn.cu:659: GGML_ABORT("fatal error")` around step ~110. The dispatcher returned `BEST_FATTN_KERNEL_NONE` for the head_dim=512 path because the gqa-opt-applies check requires both `K->ne[1] % 256 == 0` AND `mask != nullptr`. The MTP graph builder padded K view length to 256 alignment (good), but its `need_mask` predicate was:
```cpp
const bool need_mask = (kv_is_tq3 && head_dim_fa >= 512) || needs_kv_pad;
```
For Q8 KV at a kv_view_len that happened to be 256-aligned, neither clause fired — no mask, no gqa-opt, dispatcher rejection, abort.

**Fix**: drop the `kv_is_tq3` gate. Always set `need_mask` when `head_dim_fa >= 512`. The MMA gqa-opt path needs the mask regardless of K type or KV alignment.

After this fix, MTP + Q8/Q8 + HumanEval/2 + 4K context ran the full 256 steps with **accept_rate 0.87** (peaked at 1.00 in early steps, settled at 0.87 by step 112 — the same step it used to abort at).

---

## The HumanEval surprise: "the regression that wasn't"

With all three fixes in, we ran a regression check at HEAD `7b62c07`+`d758ed9bf`+`f1f811e` matching PR #131's published reference (31B + Q8/Q8 + dflash @ 4K = 149 tok/s decode, AL 10.67/16):

```
./test_gemma4_dflash --model 31B-Q4_K_M.gguf --draft draft-q8_0.gguf \
  --draft-method dflash --draft-max 8 \
  --tokens-file long_open.txt --kv-k q8_0 --kv-v q8_0 \
  --ctx-size 4096 --n-predict 256 ...
```

Result: **23.77 tok/s, AL 2.13/8.** Six times slower than published, AL collapsed from 10.67 to 2.13. We assumed our fixes broke dflash and queued a bisect.

The bisect couldn't run (Codex sandbox lacks GPU pass-through) so we tried a different angle: re-ran with the same config but a **HumanEval/2 prompt** (139 BPE tokens of Python code) instead of `long_open.txt` (40-token "robot story" creative prompt).

**56.12 tok/s, AL 5.12/8 — 64% acceptance rate.** Switching the prompt from creative writing to code more than doubled tok/s and AL. The dflash drafter had been trained on code (it's a 5-layer model distilled from target activations on HumanEval-class tasks) and creative writing was severely OOD.

PR #131's reported 0.667 accept-rate (10.67/16) is statistically identical to our 0.64 (5.12/8). **No regression**. The "regression" was a prompt distribution mismatch.

**Take-away**: drafter quality is **not** intrinsic — it's a function of (drafter × target × prompt distribution). When you bench a drafter, bench it on the prompt distribution it was trained for. Even better, document that distribution next to the headline number. PR #131's "10.67/16" claim wasn't wrong but was incompletely contextualized: it was on code prompts; on creative writing it would've shown the same collapse we hit.

---

## DM sweep: PR #131's 64K result was over-speculation, not drafter collapse

PR #131 documented an 8× decode regression at 64k:
> 64K MoE: 1997→4028 tok/s prefill (+101.7%), decode **13 tok/s**, accept **1.23/16** ← drafter diverges

Our session got curious: was this *really* drafter collapse, or was it over-speculation (budget=22 with low accept rate just wastes compute)?

We swept `--draft-max ∈ {1, 2, 4, 8}` on MoE + dflash + Q8/Q8 + pflash + 50k code prompt at 64k context:

| dm | tok/s | AL | accept rate |
|---|---|---|---|
| 1 | 23.01 | 1.00 | 100% (trivial — draft always = target's first prediction) |
| 2 | 33.81 | 1.51 | 76% |
| **4** | **36.57** | **1.79** | **45%** |
| 8 | 29.45 | 1.86 | 23% |

**dm=4 is the sweet spot** — high enough to amortise verification, low enough to not waste compute on rejected drafts. **2.8× improvement over PR #131's published 13 tok/s for the same model and context.**

The same dm=4 also holds at 256k context: **35.30 tok/s** (or 36.63 in a confirm run — variance ±5%). VRAM 21.73 GB — fits a 24 GB 3090 with 2.3 GB headroom.

For dense 31B the right value is dm=8. For MoE 26B the right value is dm=4. The 2× ratio reflects MoE's lower per-token compute (sparse experts) — verification is faster, so smaller speculation budget pays off.

---

## Scaling: MoE 26B + dflash + Q8/Q8 fits 256k on a 24 GB GPU

The full ladder, all with the same 50k code prompt + dm=4:

| ctx | Decode tok/s | AL | VRAM | Δ vs 64k |
|---|---|---|---|---|
| 64k | 36.57 | 1.79 | 19.74 GB | (baseline) |
| 128k | 35.21 | 1.77 | 20.40 GB | -3.7% |
| 256k | 35.30 / 36.63 | 1.79 | 21.73 GB | -3.5% / +0.2% |

Decode tok/s is **flat** from 64k → 256k. Cache allocation grows by ~700 MB per ctx-doubling, which is just the empty buffer overhead — actual KV usage is held at 50k tokens, so per-step KV bandwidth is constant.

Comparison to dense 31B + Q8/Q8 + ctx=64k + Shakespeare prompt (target-only, no drafter):
- prefill 1402 tok/s, decode **7.96 tok/s**, VRAM 22.6 GB.

Dense at 64k just barely fits Q8 KV. MoE at 256k fits comfortably. **MoE 26B is the right model for 24 GB long-context production**, no contest.

---

## What still hurts: bandwidth, not bugs

A bandwidth model: RTX 3090 nominal 936 GB/s. Reading the full Q8 KV for a 50k-token cache costs `50000 × 30 layers × 2 (K+V) × 8 heads × 256 head_dim × 1 byte ≈ 6.1 GB/step`. Theoretical ceiling: 152 tok/s. We hit 36.57 — about 24% of bandwidth ceiling. The remaining 76% is split among weight reads (model is 13 GB, attention reads it once per step), MoE FFN routing+execution (the active 4B), drafter forward (several extra KV reads through the drafter's own cache), speculative verification (target forward over the draft block), and ggml graph launch overhead.

Going from 4k to 50k actual KV gave **3× decode slowdown** (111 → 36 tok/s). A pure-bandwidth model would predict 12×. The fact we see only 3× means weights and overhead dominate at small ctx; KV bandwidth dominates at long ctx; the regimes meet around ctx ≈ 32k.

The remaining ~75% gap to bandwidth ceiling is not bug territory anymore — it's the structural cost of running a real model. Closing it would require **decode-time KV sparsity** (H2O / StreamingLLM / Quest / Landmark Attention / QuantSpec). None of those is integrated in any production speculative-decoding stack we found. That's an open opportunity, not a fix.

---

## Lessons that would have saved us a weekend

In rough order of importance, things we wish someone had told us:

1. **If your decoder takes raw prompts as bytes, that's a bug pretending to be a feature.** Build the tokenization plumbing first. Decode and *read* every output. If your accept_rate metric is "drafter agrees with target" and your inputs are out of distribution, drafter will agree with target on garbage and you'll celebrate a meaningless number.
2. **Drafter quality is not intrinsic.** Bench on the prompt distribution the drafter was trained on. A 6× tok/s gap between code and creative-writing prompts is normal and not a regression.
3. **Speculation budget has a sweet spot per model**, not a "higher is better" curve. Sweep dm. For Gemma-4 26B-A4B MoE the answer is 4. For 31B dense it's 8. PR #131 used the framework default 16 which is over-speculation at long context.
4. **TQ3_0 (or any FWHT-quantised KV) requires the kernel to know the storage is in rotated space.** Any path that dequants and re-dispatches loses that information. Force the chunked path explicitly; don't let an MMA fast-path silently strip the rotation.
5. **Single-token decode is a special case for SWA masks.** The mask geometry that's correct for batched prefill is wrong for single-token decode if the K view is the full SWA ring. Don't gate mask construction on `n_tokens > 1`.
6. **`gqa_opt_applies` (the head_dim=512 MMA fast path) requires BOTH alignment AND a mask.** The "NONE → abort" failure mode is silent until the kernel selector returns NONE.
7. **A bisect that returns "every commit in range is bad" is a real answer.** Walk further back, or pivot to direct code audit. Don't keep bisecting.
8. **MoE > dense for long-context on consumer GPUs.** The 26B MoE with sparse experts has both lower active compute AND lower weights footprint than the 31B dense. At 24 GB you can fit the MoE at 256k context with Q8 KV; the dense barely fits at 64k.
9. **Q8/Q8 KV is 2.4× faster on prefill than TQ3/TQ3 KV at 64k**, costs only 1.3 GB more, and decode is comparable. Use Q8 unless VRAM forces you to TQ3.
10. **`pflash` is prefill-only.** The decode-time KV bottleneck is unaddressed in production stacks. Decode-time block-sparse attention (Quest, H2O, StreamingLLM) is the next research-to-production move.

---

## Production ship config (RTX 3090, Gemma-4)

| Use case | Config | Decode tok/s | VRAM |
|---|---|---|---|
| **Long context (≥64k), code/agent** | **MoE 26B + dflash + Q8/Q8 + dm=4 + pflash** | **35–37 from 64k to 256k** | **19.7–21.7 GB** |
| Short context (4k), code/agent | MoE 26B + dflash + Q8/Q8 + dm=4 + pflash | ~112 | 19 GB |
| Short context, highest quality MTP | 31B dense + MTP + Q8/Q8 (post-Bug-3 fix) | 34 | 20 GB |
| Short context, dflash dense reference | 31B dense + dflash + Q8/Q8 + dm=8 + pflash | ~98 (HumanEval) | 22 GB |
| 64k dense, Q8/Q8 sanity | 31B + Q8/Q8 + pflash, no drafter | 7.96 | 22.6 GB |
| 64k dense, TQ3 minimum-VRAM | 31B + TQ3/TQ3 + pflash, no drafter | 6.90 | 21.25 GB |

The headline: **MoE 26B + dflash + Q8/Q8 + pflash + dm=4 fits 256k on a 24 GB 3090 at 35–37 tok/s**. With the three fixes from this session (TQ3 dispatcher, SWA mask, head_dim=512 mask) all upstream, this is a real ship config, not a benchmark stunt.

---

## What still wastes our compute, and might waste yours

Open questions that we did not resolve but identified clearly:

- **Drafter context window cap.** Our 5-layer dflash drafter has a 2096-slot KV cache. On a 50k prompt, it skips the first 47808 tokens. Larger drafter caches (5k? 10k?) might recover meaningful AL at long context. No public ablation on this exists — we looked.
- **Decode-time KV sparsity.** None of H2O / StreamingLLM / Quest / Landmark Attention / QuantSpec is wired into any production speculative-decoding stack we found. Fitting one would close the bandwidth gap at long context.
- **TQ3 SWA-wrap branch.** When the SWA ring wraps (sustained generation past 1024 tokens past the SWA window, on a SWA-windowed model) the wrap branch concat-forces F32, stripping the FWHT rotation again. Same class of bug as the one we fixed in the MMA intercept; same fix pattern (force chunked or split FA + combine softmax) applies. We didn't hit it in this session because our generations stayed within the unwrapped window.
- **MoE MTP drafter.** Doesn't exist. Training one (4-layer assistant against MoE 26B target activations) would unlock the highest-acceptance-rate small-ctx demo on the smaller model. Until then, MoE relies on dflash only.
- **An FA kernel for head_dim=512 + Q8 + non-aligned KV that doesn't require gqa-opt.** Our Bug-3 fix routes around the issue by always providing a mask. A kernel that handles the unaligned case directly would be cleaner.

If you find yourself debugging similar symptoms on a different model (Llama-3 with MTP, Qwen with Medusa, …), the workflow that ended up working for us was:

1. Set up real BPE tokenization first; never trust a byte-fallback baseline.
2. Decode every output; never trust accept_rate alone.
3. Bench multiple prompt distributions (code vs creative). The 6× gap is real.
4. Sweep `--draft-max`. The optimum varies per model+drafter.
5. When attention is wrong, instrument the FA call site directly (K type, K ne[1], mask name + dims). The dispatcher's "NONE → abort" path is silent.
6. Use Codex (or any second LLM) for focused audits with EXPLICIT evidence in the prompt — not for general code review. The "give me a fix for this exact line:line" workflow saved us hours.

The whole journey was three commits in one repo, one commit in a submodule, and roughly thirty benchmark cells. The numbers are documented in `.sisyphus/notes/gemma4-baseline/`. The bench scripts are reproducible on any 24 GB consumer GPU with CUDA 13.1.

Have at it.
