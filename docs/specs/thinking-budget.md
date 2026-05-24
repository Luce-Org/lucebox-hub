# Thinking budget — separate think vs reply token caps

A design spec for `dflash_server`'s handling of "thinking" requests:
prompts where the model is expected to produce an internal reasoning
trace before its visible reply. The spec covers the request opt-in,
the configuration surface, the two close strategies (Level 1 and
Level 2), the multi-dialect response shape, and the close-kind
taxonomy.

## 1. Background

A reasoning-capable model wraps its internal scratch work in a
delimited block — by convention `<think> … </think>` for Qwen-family
chat templates, and equivalent tags for other architectures. The
text inside is the **reasoning trace**; the text after `</think>` is
the **visible reply**.

A single combined token cap (`max_tokens` on the wire) is not enough
to control these requests:

- On hard reasoning prompts the model can spend its entire budget
  inside the `<think>` block and never emit `</think>`. The response
  arrives with no parseable answer.
- Even when the model does close `</think>` on its own, a tight cap
  can leave it with no remaining tokens to write the actual answer.

We need two independent caps — one on reasoning length and one on
the combined output — plus a server-side mechanism that *forces*
the model out of `<think>` if the reasoning cap is reached without
the model self-closing. That contract is the **thinking budget**.

## 2. Terminology

- **Phase 1 — reasoning.** Generation between the opening `<think>`
  and the model's `</think>`. Output is reasoning text.
- **Phase 2 — content.** Generation after `</think>`. Output is the
  visible reply.
- **Budget envelope.** The set of caps a thinking-enabled request
  agrees to be governed by: phase-1 cap, combined cap, and reply-
  budget reserve. See §3.
- **Close kind.** How `</think>` ended up in the stream. See §6.

## 3. Configuration

The thinking budget is **server configuration**, not a wire-protocol
field. Operators pick the policy at server start. The wire stays
standard OpenAI/Anthropic so generic clients (vLLM, llama-server,
OpenRouter passthrough, plain `curl`) work unchanged.

### Server CLI

```
dflash_server \
  --think-max-tokens 10000 \         # Phase-1 cap (default 10000)
  --default-max-tokens 16000 \       # Combined cap when the request
                                     # omits max_tokens (default 16000)
  --hard-limit-reply-budget 512      # Tokens reserved for the visible
                                     # reply when Level 2 force-closes
                                     # (default 512)
```

### Defaults

Defaults are aligned with the reference values in antirez/ds4
`ds4_eval.c` so quality numbers diff cleanly against the upstream
benchmark suite without further normalization:

| Knob | Default | Role |
|---|---|---|
| `--think-max-tokens` | 10000 | Phase-1 token cap |
| `--default-max-tokens` | 16000 | Combined (phase-1 + phase-2) cap when the request omits `max_tokens` |
| `--hard-limit-reply-budget` | 512 | Tokens reserved for the visible reply when Level 2 force-closes the reasoning block |

## 4. Request shape

A client opts into the budget envelope with `thinking:{type:"enabled"}`:

```json
{
  "model": "...",
  "messages": [...],
  "max_tokens": 16000,
  "thinking": {"type": "enabled"}
}
```

- `thinking.type` is the only field that matters; `"enabled"` opts
  in, anything else (or omitting `thinking`) keeps the legacy single-
  cap behavior.
- No `budget_tokens`, `hard_max_tokens`, or other per-request knob is
  read. The budget lives entirely in server config; clients cannot
  override it. This is intentional: it prevents silent truncation on
  backends that don't speak `thinking` (they ignore the field) and
  keeps cross-server comparisons apples-to-apples.

`reasoning:{effort: "medium"|"high"}` (the OpenAI Responses opt-in)
also turns on `<think>` rendering in the chat template, but it does
**not** activate the budget envelope. A `reasoning.effort` request
remains on the legacy single-cap behavior so existing OpenAI
Responses clients see a stable shape. Only `thinking:{type:"enabled"}`
unlocks Level 1, Level 2, and `finish_details` emission.

## 5. Close strategies

When a request opts into the budget envelope the server uses one of
two strategies to ensure the response contains a visible reply, in
order of preference. Both are independent of the model architecture
in their contract; their implementation differs per backend.

### 5.1 Level 1 — phase-2 reprompt

When the daemon finishes phase-1 generation and `</think>` did not
appear in the stream, the server constructs a fresh prompt:

```
<original prompt tokens>
<phase-1 reasoning tokens>
</think>

Final answer:
```

It then runs a second daemon call against that prompt for at most
`max_tokens − phase1_emitted` more tokens and appends the result as
the visible reply.

Level 1 works on any backend; it does not require sampling-loop
integration. Its cost is one extra prefill of the phase-1 reasoning,
which dominates for long traces.

### 5.2 Level 2 — in-process force-close

When supported by the backend (currently Qwen3.5/3.6, Gemma4, Laguna),
the server avoids the phase-2 reprompt by overriding sampling in the
generation loop:

- Track the number of tokens generated since entry to the AR loop.
- When `(n_gen − generated) ≤ --hard-limit-reply-budget`, the
  remaining headroom is dedicated to the visible reply. Override the
  next sampled token with the tokenizer's `</think>` close-tag.
- Close tags that tokenize to multiple ids (e.g. DeepSeek/Laguna,
  where `</think>` is `[1718, 37947, 32]`) are injected as a multi-
  token sequence: each subsequent loop iteration overrides one more
  token until the sequence is complete. Single-token close tags
  (Qwen3.6 `</think>` = id 248069) finish in one override.
- After the close sequence, normal sampling resumes. The model
  continues from a still-hot KV cache and writes the visible reply
  naturally, with `--hard-limit-reply-budget` tokens of headroom.

Level 2 is strictly cheaper than Level 1 (no reprompt, no second
prefill, KV cache preserved) and produces a higher-quality reply
because the model's reasoning context is still in-frame when it
writes the answer.

When a Level 2-capable backend serves a thinking-enabled request,
Level 2 fires first. Level 1 remains as a fallback for backends that
do not yet implement the BudgetHook, and for safety in case Level 2
encounters an unexpected state.

### 5.3 Budget arithmetic

In Level 2 the budget check runs against tokens **generated in the
current AR loop**, not against the absolute KV position:

```
generated = committed_now − committed_at_entry
remaining = n_gen − generated
if remaining ≤ --hard-limit-reply-budget: force-close
```

This frame matters because `committed_now` includes the prompt
length and any tokens already committed before AR took over (e.g.
when the spec-decode path tails off into AR for the final stretch).
Without the offset the check would fire `prompt_len` tokens early
and could go negative after spec-decode tail-off, force-closing
immediately as AR began.

## 6. Response shape

### 6.1 Reasoning text — multi-dialect aliases

Different reasoning-capable APIs put the reasoning trace under
different keys. There is no agreed-upon standard; each provider
picked one shape and tooling has fragmented around it.

| API | Reasoning text field | Reasoning-token count field |
|---|---|---|
| OpenAI o1/o3 | not exposed (tokens are hidden) | `usage.completion_tokens_details.reasoning_tokens` |
| Anthropic Claude | `content[]: {type:"thinking", thinking:"...", signature:"..."}` (typed block) | `usage.thinking_tokens` |
| DeepSeek R1 | `message.reasoning_content` (flat string) | inferred from totals |
| Qwen3 native | inline `<think>...</think>` in `message.content` | not exposed |
| OpenRouter | `message.reasoning` (flat) + `message.reasoning_details[]` (typed-block list) | `usage.completion_tokens_details.reasoning_tokens` |

dflash_server emits the reasoning text under **all** of the flat-
string names plus the typed-block list, and the OpenAI-shaped token
count, so any client written against any of these shapes works
without per-server remapping:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Final visible answer.",
      "reasoning_content": "Phase-1 reasoning text…",
      "reasoning":          "Phase-1 reasoning text…",
      "reasoning_details": [
        {"type": "reasoning.text", "text": "Phase-1 reasoning text…"}
      ]
    },
    "finish_reason": "stop",
    "finish_details": {
      "close_kind": "natural",
      "thinking_tokens": 8421,
      "content_tokens":  312,
      "total_tokens":    8733
    }
  }],
  "usage": {
    "prompt_tokens":     201,
    "completion_tokens": 8733,
    "total_tokens":      8934,
    "completion_tokens_details": {
      "reasoning_tokens": 8421
    }
  }
}
```

Field semantics:

- **`message.content`** — the visible reply (post-`</think>` text).
  Standard OpenAI Chat Completions field.
- **`message.reasoning_content`** — flat string with the full
  reasoning text. DeepSeek R1 convention. Primary field; tooling
  that knows only one of these field names should know this one.
- **`message.reasoning`** — same string as `reasoning_content`,
  under OpenRouter's normalized name.
- **`message.reasoning_details`** — a list of typed reasoning
  blocks. Today always exactly one `{type:"reasoning.text", text:…}`
  block carrying the full reasoning. The list shape leaves room to
  add phase-1/phase-2 splits, Anthropic-style signature fields, or
  per-stage metadata in a future version without breaking clients.
- **`usage.completion_tokens_details.reasoning_tokens`** — count of
  tokens attributed to reasoning. Matches OpenAI o1/o3's location
  and OpenRouter's normalization.
- **`finish_details`** — see §6.2.

The three `message.*` reasoning fields carry identical strings. They
are emitted together; clients should not assume they will diverge.

### 6.2 `finish_details`

When a request opts into the budget envelope, the response carries
an additional `finish_details` object alongside the standard OpenAI
`finish_reason`:

```json
"finish_details": {
  "close_kind":      "natural" | "hard",
  "thinking_tokens": <int>,
  "content_tokens":  <int>,
  "total_tokens":    <int>
}
```

- `close_kind` — see §7.
- `thinking_tokens` — tokens generated while the model was inside
  the `<think>` block. Equal to `usage.completion_tokens_details.reasoning_tokens`.
- `content_tokens` — tokens generated for the visible reply, summed
  across phase-1 (post-`</think>` if the model self-closed early)
  and phase-2 (Level 1 reprompt output).
- `total_tokens` — `thinking_tokens + content_tokens`.

`finish_reason` continues to follow OpenAI semantics
(`stop` / `length` / `tool_calls`). `finish_details` is additive:
clients that don't know about it ignore it.

`finish_details` is omitted when the request did not opt into the
budget envelope (no `thinking:{type:"enabled"}`).

## 7. Close-kind taxonomy

`finish_details.close_kind` records how the `<think>` block ended.
The current taxonomy is:

| Value | Meaning |
|---|---|
| `natural` | The model emitted `</think>` on its own, either before reaching the phase-1 cap or before Level 2 had to force-close. |
| `hard` | The phase-1 cap was reached without a model-emitted `</think>`. Either Level 2 force-closed the block in-loop (preserving KV) or Level 1 ran the phase-2 reprompt. |

A third value `soft` is reserved for a future voluntary-close
mechanism (logit-biasing the model toward `</think>` as the cap
approaches, before forcing it). Reserved so consumers can switch on
the value without an exhaustive-match warning when a future server
version adds it; not emitted today.

## 8. Streaming

Streaming responses (`stream: true`) honor the same configuration
knobs and emit the same reasoning text via the format-appropriate
SSE deltas (OpenAI `delta.reasoning_content`, Anthropic
`content_block_delta` with `thinking_delta`, OpenRouter
`delta.reasoning`).

`finish_details` is emitted in the final chunk for OpenAI Chat and
in the terminal `message_delta` event for Anthropic.

## 9. Out of scope

- **Per-request budget override.** Clients cannot tighten or relax
  the server's caps. Adding a wire field for this would re-create
  the silent-truncation footgun of letting non-budget-aware
  middleboxes drop the field.
- **Soft close-kind / soft-budget hint.** The mechanism (logit bias
  to nudge `</think>` selection before the hard cap) is sketched in
  §7 but not implemented.
- **Per-token close-info metadata.** The upstream reference exposes
  `(token_index, remaining_budget, rank)` for the close event. The
  current `finish_details` reports aggregate counts only.
- **Phase-1/phase-2 split inside `reasoning_details`.** Today the
  list always carries exactly one block. A future version may add
  per-phase blocks (`[{phase:1, …}, {phase:2, …}]`) — the typed-list
  shape was chosen specifically to allow this without breaking
  clients.
