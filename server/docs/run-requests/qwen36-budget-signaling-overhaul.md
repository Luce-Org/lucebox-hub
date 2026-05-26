# Run request: server-side budget signaling overhaul — Qwen3.6 verbose-after-close fix

**Date opened**: 2026-05-25
**Status**: Server-side. Supersedes `qwen36-hard-limit-reply-budget-bump.md` —
that was a band-aid (reserve more tokens); this addresses the root cause
(model doesn't know it's being budget-limited, so it generates noise).

## Problem statement

`hard_limit_reply_budget` is a unilateral, model-agnostic truncation: the
sampling loop intercepts the next sampled token and replaces it with
`</think>`. The model has **zero awareness** any of this is happening.
From the model's POV the close just *happens* mid-thought.

That's why we see this on bragi `qwen36-think4k`:

```
think4k accuracy by close_kind:
  hard:    7/65 = 10.8%
  natural: 23/27 = 85.2%
```

A force-closed Qwen3.6 doesn't write a terse answer in the visible area
— it **restarts the full derivation** because nothing told it the
thinking budget was the constraint. The hard-limit-bump fix
(`qwen36-hard-limit-reply-budget-bump.md`) just reserves more tokens so
that restart doesn't get truncated; this proposal fixes the underlying
disconnect.

Three layers of fix, each targeting a different failure mode:

## 1. Port `ds4_eval.c`'s soft-limit top-K peek

`ds4_eval.c:3030-3060` has a two-tier strategy we copied only half of:

- **Soft limit (default 1024)**: when budget drops below soft_limit AND
  `</think>` is already in the model's top-K candidates, accept it.
  This is a *negotiated* close — the model was about to wrap up anyway.
- **Hard limit (default 512)**: unilateral inject only after the soft
  window passed without self-close.

Today we only implement the hard tier. The model never gets a chance to
self-close gracefully in the budget-pressure window.

**Asks:**
1. Add `soft_limit_reply_budget` and `soft_limit_think_close_rank` to
   `ServerConfig` (matching ds4_eval.c defaults 1024 / 8).
2. In `qwen35_backend::do_ar_decode` (and gemma4, laguna), add a
   pre-hard-limit branch: if `remaining_budget <= soft_limit` and
   `close_token` is in the top-K of the current logits, accept it. Use
   `ggml_tensor`'s top-K helper or sort the logits row.
3. Add to `share/model_cards/_schema.json` and qwen3.6-27b sidecar.

**Expected impact**: ~10-15 cases shift from `hard`-close (11% pass) to
`natural`-close (85% pass). Net ~+10 pp on qwen3.6 `--think` bench.

## 2. Native `reasoning_effort` API end-to-end

Qwen3.6 was trained with the `low/medium/high/x-high/max` effort tiers
(sidecar declares the token budgets 4032/16128/32256/56832/81408). When
a request comes in with `reasoning.effort: high`, the model should be
told "you have high-effort thinking" via its trained mechanism — not
just have its sampler-loop budget set behind the scenes.

**Asks:**
1. Request parser maps `reasoning.effort` → corresponding sidecar tier
   value → `think_max_tokens` (already done per `ca09f64`).
2. **NEW**: chat-template injects the effort hint into the prompt
   before generation. For Qwen3.6 try the opening `<think>` tag form:

   ```
   <|im_start|>assistant
   <think>
   <!-- reasoning_budget: 16384 tokens; reply_budget: 2048 tokens -->
   ```

   The comment line consumes ~12 tokens of the budget but the model sees
   "you have 16384 tokens" in its context window. Qwen3.6 is reportedly
   responsive to such hints in instruction-following.

3. If the upstream chat template format doesn't carry it, add a
   per-model "thinking_preamble" field to the sidecar JSON. Default
   empty; Qwen3.6 fills it with the budget comment.

**Why this matters more than (1)**: even with perfect soft-limit
negotiation, if the budget is genuinely tight (e.g. 4k for a problem
that needs 8k), the model can't *plan* its reasoning to fit. Telling it
up-front lets the model self-compress.

## 3. Budget hint baked into the chat template

Make the budget signal part of every chat-template render when thinking
is enabled. This is the unification that lets all entrypoints (HTTP
API, smoke test, lucebox CLI) get budget-aware behavior for free —
nobody needs to remember to add a system message.

**Concrete proposal**:

```cpp
// In chat_template.cpp render_chat_prompt_text(), when think_mode != NONE:
// inject after <think> opening, before the model's content slot:
buf_appendf(&out, "<think>\n<!-- thinking_budget: %d tokens; "
                  "reply_budget: %d tokens. After </think>, output the "
                  "final answer line directly without restating derivation. -->\n",
            think_max_tokens, hard_limit_reply_budget);
```

Configurable knobs (sidecar):
- `thinking_preamble`: free-form string (with `{think_max}` /
  `{reply_max}` template substitutions), default empty
- `thinking_preamble_format`: `"comment" | "directive" | "none"` —
  for models where HTML-style comments confuse the tokenizer

Qwen3.6 sidecar would set `thinking_preamble_format: "comment"` (or
`"directive"` if the comment approach doesn't survive tokenization).

## Why "engine-owned" matters

> "if there's prompting necessary to tell the model about thinking
> budgets it should be part of the engine"

Every API entrypoint that hits the server today gets the same wrong
behavior — bench, lucebox CLI, /v1/chat/completions direct, /v1/messages.
Fixing it once in the chat template + sampler is a single source of
truth. Bench-side hacks would only fix the bench.

## Asks summary

| Layer | Where | Effort |
|---|---|---|
| 1. Soft-limit top-K peek | `qwen35_backend.cpp` (mirror to gemma4/laguna), `ServerConfig`, sidecar schema | Medium |
| 2. `reasoning_effort` → budget signal in template | `chat_template.cpp` render path + sidecar `thinking_preamble` field | Small-medium |
| 3. Budget hint default for Qwen3.6 | `share/model_cards/qwen3.6-27b.json` `thinking_preamble` value | Trivial |

All three are server-side. Bench stays as-is and benefits automatically.

## Validation

After landing, run on bragi (RTX 5090 MaxQ):

```bash
python dflash/scripts/bench_http_capability.py \
  --url http://127.0.0.1:8080 --area ds4-eval --model dflash \
  --think --reasoning-effort high \
  --json-out dflash/docs/tuning-snapshots/bragi-rtx5090laptop-qwen36-2026-05-26-think-with-budget-hint-full92/result.json
```

Compare against `bragi-rtx5090laptop-bragi-qwen36-think4k-ds4eval-2026-05-24/`
(30/92 = 32.6%). Target: ≥58/92 (matches OR --think 5/21 baseline).

If the engine signals budget correctly to the model, we should see:
- `close_kind=natural` ratio rise from 29% → 60%+
- Fewer `finish_reason=length` after the close (model writes terse answer)
- Per-suite AIME accuracy climb (currently 0% on think4k vs 24% on nothink)
