# Run request: bragi gemma4 + laguna config issues block local benchmarking

**Date opened**: 2026-05-25
**Status**: Bragi has gemma-4-{26b,31b} + laguna-xs.2 models downloaded
(~58 GB) and queued in the master sweep, but neither produces usable
data at default config. Detail below.

## 1. Gemma4 — chat template / thought-token leakage in visible content

Gemma4-26b-A4B-it Q4_K_M loads cleanly on bragi (RTX 5090 MaxQ 24 GB)
and responds to most prompts coherently. But specific prompts trigger
internal `<thought>` / `<channel|>` tokens to leak into the visible
content stream rather than being routed to `reasoning_content`.

Reproduced with simple curl probes (`thinking: {type: "disabled"}` set
explicitly):

| Prompt | Visible content (truncated) |
|---|---|
| `"What is 2+2?"` | `'\n4//thought\n\n4\nthought\n\n4'` |
| `"Hello, world!"` | `'\nHello! How can I help you today?\nthought\n\n Hello! How can I help you today?node_modules/lodash/package.json:\n{\n  "name": "lodash"…'` |
| `"Continue: 1, 2, 3,"` | `'\n4deviserthought\n\n4, 5, 6, 7, 8, 9, 10...\n\nthought\n\n4\nthought\n\n4'` |
| `"What color is the sky?"` | clean ✓ |
| `"Translate to French: cat"` | clean ✓ |

The "thought" string appears verbatim and the model loops, sometimes
then drifting into unrelated content (node_modules JSON). Looks like
the `<channel|>` channel-marker token (id 101 per server log) is
making it into the rendered content when the chat-template renderer
doesn't suppress it, OR the SSE emitter's per-arch token mapping
doesn't catch the gemma4 channel tokens.

Server confirms gemma4 detected:
```
[server] gguf meta: general.name='Gemma 4 26B A4B It' general.architecture='gemma4'
[server] level-2 force-close: <channel|> = 101 (1 token), hard_limit_reply_budget = 512
```

Also: `usage.timings` reports `decode_ms: 0.0`, `decode_tokens_per_sec: 0.0`,
`prefill_ms: 0.0` for gemma4 — timing instrumentation appears to be
qwen35-only, not wired through Gemma4Backend yet.

### Asks for gemma4

1. **Chat template fix**: confirm `chat_template.cpp` renders gemma4
   prompts correctly when `enable_thinking: false` is set (or
   `thinking: {type: disabled}` per Anthropic shape). The garbage
   suggests the template is sending channel markers in the prompt
   that the model then echoes back.
2. **Per-arch token mapping in SSE emitter**: the `<channel|>` open
   tag (and `<|channel>` close, or whatever the pair is) needs the
   same "consume + transition mode" treatment that qwen3.6's
   `<think>` / `</think>` got. Right now it appears to fall through
   to raw text.
3. **Gemma4Backend timings instrumentation**: mirror sindri's
   `3b80fa8` (qwen35 `usage.timings` emit) into `Gemma4Backend::generate`
   so cross-arch perf comparison is possible.

### Validation
A clean smoke after the fix: `curl -d '{"messages":[{"role":"user","content":"What is 2+2?"}],...}'`
should return content `'4'` (or short coherent answer) with NO
`thought` substring. `usage.timings.decode_tokens_per_sec` should be
nonzero.

## 2. Laguna XS.2 — OOM at default max_ctx on 24 GB VRAM

Laguna-xs.2-Q4_K_M loads 18.77 GiB of weights on RTX 5090 MaxQ
(24 GB), then immediately OOMs allocating the KV cache:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 8160.04 MiB on device 0:
   cudaMalloc failed: out of memory
alloc_tensor_range: failed to allocate CUDA0 buffer of size 8556421120
cache failed: laguna cache: ggml_backend_alloc_ctx_tensors failed
[backend_factory] LagunaBackend init failed
[server] backend creation failed
```

18.77 (model) + 8.16 (KV at max_ctx=98304, tq3_0/tq3_0) ≈ 26.93 GiB,
exceeding 24 GB.

### Asks for laguna

1. **Document max_ctx limits in the laguna sidecar**: add a
   `"recommended_max_ctx_by_vram"` table (or similar) so the
   entrypoint can derive a safe default. e.g.

   ```json
   "recommended_max_ctx_by_vram_gib": {
     "24": 32768,
     "32": 65536,
     "48": 98304
   }
   ```
2. **OR**: tune the laguna container default to use `max_ctx=32768`
   (which fits in ~2.7 GiB KV + 18.77 GiB model = 21.5 GiB, fits 24 GB)
   and bump only when the host card has more headroom.
3. **Documentation**: lucebox profile / model card README should
   mention "Laguna XS.2 (Q4_K_M) needs ≥32 GB VRAM at default 98k
   context; bragi-class 24 GB cards must lower to 32k".

## Impact

Blocking the master sweep's gemma4-26b / gemma4-31b / laguna-xs.2
columns. Until these land, lucebox's bench coverage matrix is:

| Backend | luce-dflash bragi |
|---|---|
| qwen3.6-27b ✓ | full coverage (nothink, think4k, think65k, forge) |
| gemma-4-26b   | blocked — chat template / token leakage |
| gemma-4-31b   | blocked — likely same as 26b |
| laguna-xs.2   | blocked — OOM at default max_ctx on 24 GB |

OR + vidar reference baselines exist for all four models, but the
local luce-dflash measurements that would show quant impact /
per-platform perf can't run until these are fixed.
