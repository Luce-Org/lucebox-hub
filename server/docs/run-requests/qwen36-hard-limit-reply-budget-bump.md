# Run request: bump `hard_limit_reply_budget` for Qwen3.6 (schema + sidecar)

**Date opened**: 2026-05-25
**Status**: Server-side fix. Root cause for bragi's `qwen36 think4k` collapse (30/92 = 32.6% vs nothink 51/92 = 55.4%).

## Why

A full-92 bragi `--think --max-tokens 5120` run on Qwen3.6-27B Q4_K_M
landed at **30/92 = 32.6%** — much worse than the same model in
`--no-think` (51/92 = 55.4%) and far below OR `--think` (58/92 = 63.0%).

Per-row analysis of the 60 failed cases (all `finish_reason=length` +
`close_kind=hard`):

```
think4k accuracy by close_kind:
  hard:    7/65 = 10.8%   ← force-close fires, model fails to answer in budget
  natural: 23/27 = 85.2%  ← model self-closed </think>, then answered cleanly
```

A representative failed case (`recNu3MXkvWUzHZr9`, GPQA Diamond):

- `thinking_tokens=4597`, `content_tokens=523`, total=5120 → max_tokens hit
- `reasoning_content` ends with **"Final decision: B."** (correct answer)
- `content` is the model **restarting the full derivation in the visible
  area** — it writes "To find the time experienced by the astronaut, we…"
  and runs out of tokens mid-derivation, never reaching the "Answer: B"
  line. Bench grades `given=C` (random extraction from partial work)
  even though the model decided correctly.

Qwen3.6 after `</think>` is verbose: it re-states its work in the
visible area before writing the answer line. 512 tokens of reply room
is not enough for that pattern. Two of three cases in the failed bucket
need ~1.0-1.5k tokens to complete the post-close answer.

## The 512 value's origin

`hard_limit_reply_budget = 512` is the default in `ds4_eval.c:1528`
(antirez's reference for DeepSeek-V4-flash, a terse "Answer: N" model).
We inherited it as the global default and never overrode it for
Qwen3.6. The C++ `model_card.cpp:310` already supports per-model
override (`card.hard_limit_reply_budget`), but no sidecar declares it
and the schema doesn't list the field, so all models fall back to 512.

## Asks

### 1. Schema addition

`share/model_cards/_schema.json` — add the optional field:

```jsonc
"hard_limit_reply_budget": {
  "type": "integer",
  "minimum": 256,
  "description": "Tokens reserved post-`</think>` for the visible answer phase. Per ds4_eval.c reference. Default 512 (terse models). Bump for verbose models that restate work after force-close (Qwen3.6 wants ~2-4k)."
}
```

### 2. Sidecar updates

| Model | Suggested `hard_limit_reply_budget` | Reasoning |
|---|---|---|
| `qwen3.6-27b.json` | **4096** | verbose post-`</think>`; per-row data shows ~500-1500 tokens used after close, 4k leaves safe headroom |
| `gemma-4-26b-a4b-it.json` | **2048** | medium-verbose; only +1 to +3 net think delta on OR suggests less verbose than qwen but still wants more than 512 |
| `gemma-4-31b-it.json` | **2048** | same family |
| `laguna-xs.2.json` | **512** (default) | code model, near-zero think delta, no need to bump |

With `qwen3.6-27b.json max_tokens=32768` and reply_budget=4096, the
derived `think_max_tokens` becomes 28672 — still plenty of thinking
room for the model's reasoning.

### 3. (optional) Reply-budget defaults in family fallback

`model_card.cpp` family-fallback table for `qwen35` arch should also
bump from 512 → 4096 so users who haven't yet mounted the sidecar
volume don't hit the same trap.

## Output

After the schema + sidecar updates, a single confirmatory run on RTX
5090 MaxQ (bragi) or RTX 3090 Ti (sindri):

```bash
.venv/bin/python dflash/scripts/bench_http_capability.py \
  --url http://127.0.0.1:8080 \
  --area ds4-eval \
  --model dflash \
  --think \
  --max-tokens 32768 \
  --json-out dflash/docs/tuning-snapshots/<host>-qwen36-2026-05-25-think-replyhint-full92/result.json
```

Compare against the failed `qwen36-think4k-ds4eval` baseline. Expected:
the ~30 cases that previously force-closed and ran out of reply budget
should now have room to write the answer line, lifting accuracy from
32.6% closer to the OR reference (58/92 = 63%).

## Out of scope

This run-request does NOT propose changing post-`</think>` scaffolding
(injecting "Final answer:" text after force-close) — that was explicitly
removed in `ca09f64` because it leaked into clean-close responses. The
reply-budget bump is the surgical fix; if accuracy still trails after
the bump, scaffolding is a follow-up question.
