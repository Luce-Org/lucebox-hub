# PFlash Drafter Unification — Experiment Plan

## Goal

Reduce VRAM and simplify config by removing one of the four resident
weight sets in the dual-speculator server:

```
Decode target (Qwen3.6-27B-Q4_K_M)              15.3 GB   (load-bearing)
DFlash spec-decode drafter (z-lab Q4 GGUF)       1.0 GB   ── duplicate work?
MTP NextN heads (in target gguf)                 0.5 GB   (load-bearing)
PFlash compress drafter (Qwen3-0.6B-BF16)        1.5 GB   ── replaceable?
                                                ─────
                                                 18.3 GB  + KV ~1.5-3 GB
```

The two drafters serve different functions:
- DFlash decode drafter: predict next tokens in target's vocab; rich
  feature-alignment to target hidden states; small context (~4K SWA).
- PFlash compress drafter: per-token importance scoring via Block-Sparse
  Attention; long context (32K-131K); produces top-k indices not logits.

## Hypothesis space

```
H1: PFlash scorer is model-agnostic.
    → swapping Qwen3-0.6B → Qwen3.5-0.8B preserves NIAH recall + compression ratio.
    → smallest first cut. Already supported by the existing loader.

H2: PFlash needs a drafter from the SAME tokenizer family as the target.
    → cross-family drafters (TinyLlama) collapse recall on Qwen prompts.
    → useful sanity bound but not a goal.

H3: One drafter can serve BOTH roles if it has BSA + long-ctx + target
    vocab alignment.
    → e.g. dflash-draft-3.6 with a BSA path + SWA extension to 32K.
    → biggest win (saves 1.5 GB, perfect tokenizer match).
    → highest engineering cost.

H4: The PFlash drafter's compress-time KV cache can prime decode-time KV
    when the same drafter is used.
    → only meaningful if H3 succeeds.
    → could shave seconds off TTFT on multi-turn with shared prefix.
```

## Experiments (cheapest first)

### E1 — Qwen3.5-0.8B-Q4 as PFlash drafter — **RAN; FAILED (instructive)**

Result: 3/3 NIAH MISS at 32K/65K/131K. Server log: `[compress] drafter
init failed: missing top-level tensors (token_embd/output_norm/output)`.

Root cause: the only Qwen3.5-0.8B GGUF on disk
(`/home/peppi/models/qwen3.5-0.8b-draft/Qwen3.5-0.8B-Q4_K_M.gguf`, 508 MB)
is a **DFlash draft variant** — `lm_head` and `token_embd` are stripped
because DFlash drafts replace the vocab projection with a feature
projection trained against the target's hidden states. PFlash loader
requires the full embed/output for attention scoring.

The load-bearing finding for the whole unification idea:

> **A DFlash draft model is structurally incompatible with being a
> PFlash drafter.** Same arch family, different weight set: PFlash
> needs full embed+output for BSA scoring; DFlash drafts strip those
> on purpose (the whole point of being a draft is feature alignment
> to target hidden states, not standalone next-token prediction).

So Path D (one model both jobs) is **not feasible with DFlash drafts**.
Two viable shapes remain:

1. **Use a complete base model as BOTH PFlash scorer AND DFlash drafter**.
   Drops DFlash's feature-alignment training advantage; DFlash accept
   would fall (essentially target-only logits on a smaller model).
2. **Two drafters that share lower layers** — separate output heads
   for PFlash (vocab projection) and DFlash (feature projection), shared
   backbone. Requires retraining. Significant effort.

To actually run E1 we need a complete Qwen3.5-0.8B base GGUF (not on
disk; ~1.5 GB download from HF). Until that's downloaded, E1 stalls.

#### E1 second attempt — downloaded `unsloth/Qwen3.5-0.8B-GGUF`

Pulled both BF16 (1.5 GB) and Q4_K_M (508 MB) variants. NIAH sweep
failed again with the same `missing top-level tensors` error.

Inspection of the new BF16 GGUF showed:
- `general.architecture = qwen35` (hybrid arch with SSM blocks, NOT the
  attention-only qwen3 arch the PFlash drafter graph was written for)
- `token_embd.weight` ✓ shape [1024, 248320] (tied to lm_head)
- `output_norm.weight` ✓ shape [1024]
- `output.weight` **missing** (lm_head ties to token_embd — standard for
  small models)
- Block tensors: `attn_qkv` (fused), `attn_gate`, `ssm_alpha`,
  `ssm_beta`, `ssm_conv1d`, `ssm_dt.bias`, `ssm_norm`, `ssm_out`,
  `attn_norm`, `post_attention_norm`, `ffn_*` — i.e. the same hybrid
  SSM+attention layout as the 27B target.

Routing trace: `load_drafter(arch=qwen35-0.8b)` dispatches in
`qwen3_drafter.cpp:153` to `load_target_gguf()` — the 27B target loader
— which insists on `output.weight` being present (line 442 of
`gguf_target_loader.cpp`). Hard error.

So the path is blocked by TWO things, not one:

1. **`output.weight` required even when embeddings are tied.** Trivial
   fix (~10 lines): alias `out.output = out.tok_embd` when
   `output.weight` is missing.
2. **`load_target_gguf` was sized for the 27B's tensor layout.** Reading
   the rest of the loader: layer wiring, KV cache sizing, FA window
   defaults — all assume the 27B's `n_layer=64`, `n_embd=5120`,
   `n_head=24`, `head_dim=256`. Loading the 0.8B model would also
   require this loader to read its dims from the GGUF rather than
   inheriting from compile-time constants. **Probably already works**
   because `n_layer`/`n_embd`/etc. ARE read from the GGUF metadata in
   the load path — but the graph builder downstream might assume
   constants. Needs a smoke test.

Effort to make E1 actually run: small (~30 lines of loader work +
maybe a graph-builder constant audit). NOT done in this session.
**Documented as a follow-up; not on the critical path for shipping the
dual-speculator branch.**

The deeper conclusion stands: even with the loader fix, this only buys
us "use a smaller PFlash drafter" — which is a 1 GB VRAM win, not an
architectural shift. The original Path D goal (one drafter, two jobs)
still requires DFlash retraining or a multi-head model.

### E1' — Original wiring + reproduction notes

Status: WIRING DONE on `feat/mtp-prefix-warm-ghost`:
- `_prefill_hook.py`: added `--prefill-drafter-arch` CLI flag and
  `PrefillConfig.drafter_arch` field; arch propagates through to the
  daemon compress command.
- `qwen35_backend.cpp::handle_compress`: parses an optional `[drafter_arch]`
  token between drafter path and `nopark`; calls
  `load_drafter(path, gpu_layers, arch, ctx)` with the parsed arch.
- Build passes; existing handle_compress callers (3-arg form, plus
  3-arg + nopark) unchanged.

Command:
```
python3 dflash/scripts/server.py … \
  --prefill-drafter /home/peppi/models/qwen3.5-0.8b-draft/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prefill-drafter-arch qwen35-0.8b \
  --prefill-drafter-tokenizer Qwen/Qwen3.5-0.8B   # if tokenizer matters; otherwise omit
```

Measure: NIAH recall @ 32K/65K/131K, compress wall time, compressed token
count (compression ratio), then decode tok/s + accept via MTP path.

Compare to baseline (Qwen3-0.6B-BF16, captured earlier today):

| size  | baseline compress | baseline ratio | baseline NIAH |
|-------|-------------------|----------------|---------------|
| 32K   | 6.7 s             | 20.4×          | ✅            |
| 65K   | 48.4 s            | 20.1×          | ✅            |
| 131K  | 83.8 s            | 20.0×          | ✅            |

Decision rule:
- If Qwen3.5-0.8B-Q4 NIAH @ 131K passes AND compress ≤ 1.5× baseline
  → ship as default (1 GB VRAM win)
- If NIAH fails OR compress > 2× baseline → keep Qwen3-0.6B-BF16, file
  finding, move to E2.

### E2 — Q4 drafter for PFlash  (~15 min, if E1 passes)

Test if the BF16 baseline buys us anything. If Qwen3-0.6B-Q4 (no such
file on disk yet; could quantize via `convert_hf_to_gguf.py --outtype
q4_k_m`) also passes NIAH, we save another ~0.7 GB on BF16→Q4.

### E3 — Path D: dflash drafter as PFlash compress (HIGH effort)

Requires:
- Add a Block-Sparse Attention forward path to dflash-draft-3.6's graph.
  Existing BSA code lives in `dflash/src/qwen3/qwen3_graph.cpp` (Qwen3-0.6B
  specific). Templating it for Block-Diffusion architecture is the bulk
  of the work.
- Extend the drafter's max context. Current ~4K with SWA. Need 32K+. Likely
  means storing per-layer compressed KV during BSA scoring (RoPE position
  encoding past trained range is the main risk).
- Verify importance scores remain useful for top-k selection. The dflash
  drafter is trained for next-token prediction, not importance ranking —
  its attention pattern may be different.

Acceptance criteria:
- NIAH @ 131K recall ≥ baseline.
- Compress time ≤ 2× baseline (likely slower because larger model).
- Drafter KV cache from compress is reusable for the subsequent DFlash
  decode pass — measure prefix-cache-style TTFT savings.

If all three pass → drop Qwen3-0.6B entirely. Save 1.5 GB + tokenizer
alignment + one place to upgrade.

### E4 — Cross-family sanity check (P3 of H2)

TinyLlama-1.1B or any non-Qwen drafter at the same size. NIAH should
fail or recall should be dramatically worse. Confirms tokenizer family
alignment matters and the unified-drafter idea only works WITHIN
Qwen3/Qwen3.5/Qwen3.6.

Effort: another 15-min run if a non-Qwen GGUF is on disk. Not on disk
right now; skip unless someone fetches one.

## Decision tree

```
E1 result → ?
├── Qwen3.5-0.8B passes
│   └── Update server default to use it
│       └── Question becomes: keep Qwen3-0.6B as fallback or remove?
│           → Remove if E1 is decisive across 32K/65K/131K
└── Qwen3.5-0.8B fails
    └── Run E2 (Qwen3-0.6B-Q4 to save BF16 → Q4 quant size)
        └── Pivot to E3 (Path D) — bigger lift but bigger payoff

E3 result → ?
├── dflash drafter + BSA path works
│   └── Drop Qwen3-0.6B; one drafter serves both jobs
│       └── Investigate H4 (KV reuse) as follow-up
└── dflash drafter + BSA doesn't converge
    └── Stop, accept the 1.5 GB cost of separate compressor
        Document the architecture constraint as known.
```

## What this does NOT solve

- The DFlash × PFlash decode-accept conflict is independent of the
  drafter choice for compress. Even with E3 working, DFlash spec decode
  still collapses to 0.14-0.21 accept on PFlash-compressed prompts
  because the issue is the *compressed token stream as decode input*,
  not the compressor identity. The auto-select route to MTP at
  prompt_tokens > 4K stays the right answer.

## Next action

E1 is wired and ready. Server is currently down. Bring it up with the
0.8B drafter and rerun the NIAH sweep — direct comparison to today's
PFlash-MTP-TQ3 data is one bench loop away.
