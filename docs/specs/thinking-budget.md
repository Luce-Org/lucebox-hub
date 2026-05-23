# Thinking budget — separate think vs reply token caps

Status: **v2 shipped (non-streaming only) — server-side config.**
v1 (per-request JSON envelope) was deprecated and removed; see "v1
deprecation" below. v3 covers streaming.

## v2 design — server-side configuration

The thinking budget is **server config, not wire protocol**. Operators
choose the policy at server start; clients send a standard OpenAI request
with no custom JSON. This restores compatibility with generic
OpenAI-compat clients (vLLM, llama-server, OpenRouter passthrough) and
removes the silent-truncation footgun that v1's custom envelope created
against backends that ignored it.

### Server CLI (matches `antirez/ds4 ds4_eval.c` knob names)

```
dflash-server \
  --think-max-tokens 10000 \      # cap on phase-1 reasoning generation
  --default-max-tokens 16000      # combined cap when request omits max_tokens
```

### Wire shape (now standard OpenAI)

```json
{
  "model": "...",
  "messages": [...],
  "max_tokens": 16000,
  "thinking": {"type": "enabled"}
}
```

`thinking.type` is the only knob on the wire: `"enabled"` opts the
request into the server's configured thinking-budget policy;
`"disabled"` (or omitting the field) falls back to single-cap behavior.
**There is no `budget_tokens` or `hard_max_tokens` on the wire.**
Backends that don't speak `thinking` (ds4_server, OpenRouter, vLLM)
ignore the field and apply `max_tokens` as a single cap, which is
exactly what we want for cross-server comparisons.

### Server enforcement (unchanged from v1 mechanism)

When the request opts in:

1. **Phase 1 — reasoning.** Daemon generates up to
   `min(--think-max-tokens, max_tokens)` tokens.
2. **Inspect.** If `</think>` appeared, parse as usual. If not (and the
   prompt opened a `<think>` block), trigger phase 2.
3. **Phase 2 — content.** Build a new prompt = original + phase-1
   reasoning + `</think>\n\nFinal answer: `, generate up to
   `max_tokens - phase1_emitted` tokens. Append to response as
   `content`.

`finish_details` (`close_kind`, `thinking_tokens`, `content_tokens`,
`total_tokens`) still appears in the response when the request opted in
to the budget envelope.

## Response shape — multi-dialect aliasing

The reasoning text is the same content under every reasoning-capable
API; different providers just chose different field names. dflash emits
the DeepSeek-style field as the primary (because that's what dflash has
emitted since v1 and what existing Qwen/DeepSeek tooling expects), plus
the OpenRouter / Anthropic / OpenAI aliases for cross-provider
compatibility.

### Comparison across providers

| API | Reasoning text field | Reasoning token count |
|---|---|---|
| OpenAI o1/o3 | *not exposed* (hidden tokens) | `usage.completion_tokens_details.reasoning_tokens` |
| Anthropic Claude | `content[]: {type:"thinking", thinking:"...", signature:"..."}` (block) | `usage.thinking_tokens` |
| DeepSeek R1 | `message.reasoning_content` (flat string) | inferred from totals |
| Qwen3 native | inline `<think>...</think>` in `message.content` | not exposed |
| OpenRouter | `message.reasoning` (flat) + `message.reasoning_details[]` (structured) | `usage.completion_tokens_details.reasoning_tokens` |
| **dflash (planned)** | `message.reasoning_content` (primary) + `message.reasoning` (alias) + `message.reasoning_details[]` (structured) | `usage.completion_tokens_details.reasoning_tokens` + dflash-specific `finish_details.thinking_tokens` |

OpenRouter is the broadest aggregator (routes to DeepSeek, Anthropic,
OpenAI-shape providers like DeepInfra/Chutes, plus self-served models)
so they had to define a normalized shape that catches everything below.
We adopt their normalization as the secondary shape on top of our
existing DeepSeek-style primary.

### Field emissions (non-streaming `/v1/chat/completions`)

When the request opted into thinking and the model produced reasoning,
dflash emits **all of the following** in the response, carrying the
same reasoning text under different keys for client-side compatibility:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Final visible answer.",
      "reasoning_content": "Phase-1 reasoning text...",
      "reasoning":          "Phase-1 reasoning text...",
      "reasoning_details": [
        {"type": "reasoning.text", "text": "Phase-1 reasoning text..."}
      ]
    },
    "finish_reason": "stop",
    "finish_details": {
      "close_kind": "natural",
      "thinking_tokens": 8421,
      "content_tokens": 312,
      "total_tokens": 8733
    }
  }],
  "usage": {
    "prompt_tokens": 201,
    "completion_tokens": 8733,
    "total_tokens": 8934,
    "completion_tokens_details": {
      "reasoning_tokens": 8421
    }
  }
}
```

Field-by-field notes:

- **`message.reasoning_content`** — stays the primary. DeepSeek R1
  shape; dflash's existing field. Qwen-gateway and DeepSeek-aware
  tooling reads this.
- **`message.reasoning`** — alias of `reasoning_content`. Same string.
  Catches OpenRouter SDKs, generic gateway clients, and the bench's
  fallback chain.
- **`message.reasoning_details`** — list of typed reasoning blocks.
  Today always exactly one `{type: "reasoning.text", text: ...}` block
  with the full reasoning. The list shape is intentional: when dflash
  exposes phase-1 + phase-2 separately in a future version, the same
  field carries `[{phase:1, type:"reasoning.text", text:...}, {phase:2, type:"reasoning.text", text:...}]`
  without breaking clients.
- **`usage.completion_tokens_details.reasoning_tokens`** — OpenAI o1/o3
  standard location, also OpenRouter's normalized shape. Mirrors the
  same count as `finish_details.thinking_tokens`. The two are kept in
  sync and emitted together.
- **`finish_details`** — stays. It's a dflash-specific operator-visibility
  extension that carries *why* thinking closed (`natural` / `soft` /
  `hard`), which no provider above us exposes. Useful for diagnosing
  long-think failures and budget exhaustion.

### Why alias rather than replace

`reasoning_content` is what the existing dflash impl emits, what the
existing bench reads as primary, and what DeepSeek-format tooling
expects. Replacing it with `reasoning` would break DeepSeek-format
clients. Emitting both is cheap (~15 LOC in `http_server.cpp` to alias
the same string under multiple keys; a few more to plumb
`reasoning_tokens` into the existing usage block) and unlocks a much
wider client surface:

- OpenRouter SDKs and apps reading `message.reasoning` work directly.
- DeepSeek/Qwen tooling reading `message.reasoning_content` keeps
  working.
- Anthropic-block-aware tooling walking `message.reasoning_details` for
  typed blocks works.
- OpenAI billing/quota dashboards reading
  `usage.completion_tokens_details.reasoning_tokens` see real numbers.

### Bench fallback chain

`bench_http_capability.py` should read reasoning text from the response
in priority order, taking the first non-empty match:

```python
reasoning = (
    msg.get("reasoning_content")                  # DeepSeek / dflash primary
    or msg.get("reasoning")                       # OpenRouter / dflash alias
    or "\n".join(
        d.get("text", "") for d in msg.get("reasoning_details", [])
        if isinstance(d, dict) and d.get("type") == "reasoning.text"
    )                                              # Anthropic-block / OR-structured / future dflash multi-phase
    or ""
)
```

This makes cross-server runs (sindri vs vidar vs OpenRouter vs an
upstream-format Anthropic endpoint) directly comparable — same
extraction code, just different field-name routing per backend.
Observed motivation: a 2026-05-23 OR run against `qwen/qwen3.6-27b`
emitted ~4000 reasoning tokens that the bench discarded as
`reasoning_content=""` because OR puts them under `message.reasoning`.
The fallback recovers them.

### Implementation status

- **`reasoning_content` primary** — shipped (v1, kept in v2).
- **`reasoning` alias** — planned, not yet emitted.
- **`reasoning_details` list** — planned, not yet emitted.
- **`completion_tokens_details.reasoning_tokens`** — planned, not yet
  emitted (count is already tracked internally as `phase1_tokens` /
  `finish_details.thinking_tokens`; just needs surfacing into the usage
  block).
- **Bench fallback chain** — planned; one-function patch in
  `bench_http_capability.py`. Lands independently of the server change.

## Why the redesign

v1 (per-request `thinking.budget_tokens` + `hard_max_tokens` JSON
fields) was a one-off protocol that no other server implemented. When
the bench pointed at vidar (`ds4_server.c`) or OpenRouter, those servers
silently ignored the custom fields and applied only the combined
`max_tokens` cap — which was set to `REPLY_MAX_TOKENS=4096` (post-think
content cap, not total). The result was every cross-server run was
truncated mid-reasoning, producing artificially-low pass rates and
hiding the truncation behind `format=False` rows.

`antirez/ds4 ds4_server.c` itself has **no** thinking-budget mechanism
— no CLI flags, no `#define`s, no in-loop force-close. The 4-knob
design (`think_max_tokens` / `max_tokens` / `soft_limit_reply_budget` /
`hard_limit_reply_budget`) lives entirely in `ds4_eval.c`'s in-process
sampling loop, which calls `ds4_session_sample()` token-by-token and
force-injects `</think>` when the budget runs out. There is no upstream
wire-format precedent to copy because ds4 doesn't expose the budget
over HTTP at all.

So v2 keeps the wire OpenAI-compatible, moves the budget choice to
server config (where it logically belongs — it's an operator decision
about how the server should behave), and frees the bench to talk to any
backend uniformly.

## v1 deprecation (removed)

- `ThinkingConfig.budget_tokens` — removed; server reads
  `--think-max-tokens` instead. Extra keys on `thinking` are ignored.
- `ChatRequest.hard_max_tokens` — removed; the combined cap is
  `max_tokens` (or `--default-max-tokens` when omitted).
- Bench constants `THINK_MAX_TOKENS` / `REPLY_MAX_TOKENS` /
  `HARD_MAX_TOKENS` — collapsed to a single `DS4_EVAL_MAX_TOKENS=16000`
  in `bench_ds4_eval.py`.

## v1 spec (historical)

Status: **shipped, then deprecated in v2.** Original content preserved
below for context on the original problem framing and the (now-removed)
JSON envelope.

### Problem

`/v1/chat/completions` had a single `max_tokens` cap that covered both the
`<think>...</think>` reasoning phase and the visible-content reply. Qwen3.6
routinely keeps thinking past `</think>` on math/code prompts and either
(a) exhausts the budget mid-derivation without ever closing thinking, or
(b) closes thinking late and runs out of tokens before writing the answer.

In our ds4-eval profile run (`quality.ds4_eval`), the two failure shapes
manifest as `format=False given=?` rows: the model thought for the entire
budget and never produced a parseable "Answer: N" line. `find_answer_with_fallback`
in `bench_http_capability.py` already scans `reasoning_content` for an
answer pattern, but if the model never even wrote a conclusion (just trailed
off mid-thought), there is nothing to extract.

### Upstream reference: antirez/ds4 `ds4_eval.c`

| Parameter | Default | Role |
|---|---|---|
| `think_max_tokens` | 10000 | thinking-block budget |
| `max_tokens` | 4096 | visible-output budget |
| `soft_limit_reply_budget` | 8192 | total tokens before hinting model to wrap up |
| `hard_limit_reply_budget` | 16384 | total tokens force-close |

Three closure modes via `eval_think_close_kind` enum: `NATURAL` (model
emits `</think>` on its own), `SOFT` (hint at soft budget), `HARD`
(force-inject `</think>` at hard budget). `eval_think_close_info` carries
`{kind, token_index, remaining_budget, rank}` so post-hoc you can tell
*why* a turn ended.

### v1 scope (non-streaming `/v1/chat/completions` only)

### Request schema additions

```python
class ThinkingConfig(BaseModel):
    type: Literal["enabled"] = "enabled"
    budget_tokens: int = 10000   # = ds4 think_max_tokens

class ChatRequest(BaseModel):
    ...
    thinking: ThinkingConfig | None = None
    hard_max_tokens: int | None = None   # = ds4 hard_limit_reply_budget
```

`max_tokens` (and the newer `max_completion_tokens`) keep their existing
meaning. When `thinking` is set, they cap the *visible content* after
`</think>`. When `thinking` is `None`, the legacy single-cap behavior
applies — no separate budgets, no re-prompt.

### Server enforcement (non-streaming path)

1. **Phase 1 — reasoning.** Run the daemon with
   `gen_len = thinking.budget_tokens`. Daemon streams up to that many
   tokens and stops.
2. **Inspect.** Decode collected tokens. Did `</think>` appear?
   - **Yes (`close_kind = "natural"`):** parse as usual. Reasoning before
     `</think>` becomes `reasoning_content`; text after becomes `content`.
     If the content section is empty and we still have room under
     `hard_max_tokens`, fall through to phase 2.
   - **No (`close_kind = "hard"`):** the model exhausted the thinking
     budget without producing an answer. Go to phase 2.
3. **Phase 2 — content (only when needed).** Build a new prompt by
   concatenating the original prompt tokens + the phase-1 reasoning text
   + a closing `</think>\n` + a brief "Final answer: " hint. Write the
   new `.bin`, send a fresh daemon command, collect up to
   `max_completion_tokens` (or `max_tokens`) tokens. Append those to
   the response as `content`.
4. **Cap.** `hard_max_tokens`, if set, governs the total of phase 1 +
   phase 2 token output; phase 2 is shortened to fit.

### Response shape

Add a `finish_details` object alongside `finish_reason`:

```json
{
  "finish_reason": "stop" | "length" | "tool_calls",
  "finish_details": {
    "close_kind": "natural" | "hard",
    "thinking_tokens": 8421,
    "content_tokens": 312,
    "total_tokens": 8733
  }
}
```

`finish_reason` stays OpenAI-compatible; `finish_details` is a custom
extension that downstream tooling (bench, snapshot) can read.

### Bench wiring

`bench_http_capability.py` exports three generic split-budget constants
named for what they do, not where the values came from:

```python
THINK_MAX_TOKENS = 10000   # thinking.budget_tokens
REPLY_MAX_TOKENS = 4096    # max_tokens (post-think content cap)
HARD_MAX_TOKENS  = 16384   # hard_max_tokens (combined cap)
```

The values happen to match `antirez/ds4 ds4_eval.c` defaults so our
cross-machine quality numbers diff cleanly against published ds4 runs,
but the mechanism applies to any thinking-enabled area (`ds4-eval`,
`all`). Per-case overrides win when the fixture sets them.

Per-case trace records `close_kind`, `thinking_tokens`, `content_tokens`
so failures can be diagnosed without re-running the model.

The legacy single-cap default (`DS4_EVAL_MAX_TOKENS = 16000`) was
removed; thinking-enabled cases unconditionally use the split budget,
non-thinking cases (smoke MC, short recall) keep their original 512
default which is just `REPLY_MAX_TOKENS`-equivalent for short answers.

### v1 deferred (TODO for v2)

- **Streaming path.** The `stream=True` SSE handler needs the same
  budget enforcement but inside the existing async token loop. Phase 2
  re-prompt is harder because we have to flush the existing stream
  cleanly before re-opening the daemon. The bench's ds4-eval invocation
  uses `stream=False`, so this is not on the critical path for the
  initial ship.
- **Soft hint (`SOFT` close mode).** ds4 nudges the model toward
  closing thinking when the soft budget is hit. Doing this cleanly
  requires either logit biasing (daemon-side C++ change) or mid-stream
  prompt injection (re-tokenize and continue). v1 reports only
  `natural` and `hard` close modes.
- **Per-turn `eval_think_close_info` rank / token_index.** ds4 records
  exactly *which* sampled token triggered close. v1 records aggregate
  counts only.

### Test plan

- Unit: `test_server.py` mocks a daemon stream that never emits `</think>`
  and asserts phase 2 fires with the expected synthetic prompt.
- Unit: thinking budget honored even when `max_tokens` is much larger.
- Integration (in the ds4-eval run): `format=False` rate should drop
  meaningfully; `close_kind="hard"` should appear in trace rows that
  previously showed `given=?`.

### Out of scope

- `/v1/messages` (Anthropic Messages API): same plumbing pattern but
  separate request schema; pick up in a follow-up once v1 is settled.
- Tool-call interactions with phase 2 re-prompt: if the model emits a
  tool call inside thinking, phase 2 currently still runs and appends
  more content. The tool-buffer interaction is left as a known edge case;
  the bench cases that exercise tool calls (`agentic-tools`,
  `agentic-session`) don't enable `thinking` so they're unaffected.
