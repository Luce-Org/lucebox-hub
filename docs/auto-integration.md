# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-27T20:26:50Z
Current HEAD: `92a324e` (`merge pr-285: feat(lucebox): docker stack + CLI + bench/profile + harness + luce-bench in-tree`)

## Included in the current stack

| PR | Head branch | State | Notes |
|---:|---|---|---|
| #152 | `main` | included | Open upstream PR still in stack; retained as-is. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | included | Draft FP16 safetensors support. |
| #266 | `feat/harness-typed-adapters` | included | Typed harness adapters + session proxy. |
| #274 | `feat/pflash-drafter-ee7` | included | pFlash early-exit drafter work. |
| #275 | `feat/pflash-drafter-ee3-default` | included | Depends on #274. |
| #276 | `fix/qwen36-claude-code-tool-calling` | included | Tool-calling fix on Anthropic path. |
| #284 | `fix/draft-safetensors-rope-theta` | included | Clean / current head included. |
| #285 | `feat/lucebox-docker` | included this run | Clean merge from current head `47ff8e7`. |
| #287 | `feat/gemma4-timings` | included | Prefill/decode timing plumbing. |
| #288 | `fix/laguna-chat-template` | included | Chat template + `eos_chat_id` fallback. |

## Attempted this run

| PR | Outcome | Notes |
|---:|---|---|
| #285 | merged cleanly | Integrated into `auto-integration` with no conflict resolution needed. |

## Held / not yet included

| PR | Head branch | State | Why held |
|---:|---|---|---|
| #278 | `fix-pflash-drafter-backend-precision-submit` | DIRTY | Draft branch; currently not stack-compatible enough to pull in without extra triage. |
| #273 | `feat-cpp-server-gemma4-layer-split-adapter` | DIRTY | Large Gemma4 adapter piece; still blocked relative to current stack shape. |
| #265 | `feat-cpp-server-target-layer-split-prep` | DIRTY | Prep work only; not yet worth stacking ahead of the current line. |
| #249 | `dynamic_prefill` | DIRTY | Older pFlash tuning work; remains held. |
| #237 | `feat/dflash-mtp-foundation` | DIRTY | Valuable, but not yet integrated against the current stack. |
| #221 | `feat/mtp-prefix-warm-ghost` | DIRTY | Older MTP/prefix-cache line; held. |
| #193 | `feature/gemma4-feature-complete` | DIRTY | Very large/old Gemma4 stack; likely superseded by newer pieces. |
| #183 | `split/gemma4-11a-target-mtp-integration` | DIRTY | Older Gemma4 split chain; held pending dependency review. |
| #182 | `split/gemma4-10-mtp-loader-step-graph` | DIRTY | Older Gemma4 split chain; held pending dependency review. |
| #181 | `split/gemma4-09-dflash-draft-runtime` | DIRTY | Older Gemma4 split chain; held pending dependency review. |
| #180 | `split/gemma4-08-draft-loader-quant` | DIRTY | Older Gemma4 split chain; held pending dependency review. |
| #177 | `split/gemma4-06-kv-correctness` | DIRTY | Older Gemma4 split chain; held pending dependency review. |
| #174 | `split/gemma4-14-small-vram-docs` | DIRTY | Docs-only older Gemma4 split chain; held with the rest of the stack. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | DIRTY | Older dFlash MTP line; held. |
| #153 | `xabicasa/dflash-mtp-integrated` | DIRTY | Older dFlash MTP line; held. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | DIRTY | Older build/config line; held. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | DIRTY | Older scheduler work; held. |
| #131 | `feature/gemma4-support` | DIRTY | Broad, older Gemma4 support stack; held. |
| #94 | `feat/dflash-qwen36-swa-draft` | DIRTY | Oldest held branch in the current triage set. |

## Conflicts fixed

- None in this run. PR #285 merged cleanly with `ort`.

## Validation run

- `PYTHONPATH=src uv run --python 3.12 --with pytest python -m pytest tests/test_runner.py tests/test_smoke_area.py tests/test_smoke_end_to_end.py`
  - Result: `43 passed`

## Helper branches

- None created this run.

## Notes

- The repository already contains earlier integration commits for #142, #152, #266, #274, #275, #276, #284, #287, and #288.
- PR #285 had advanced since the last recorded integration attempt; this run pulled in its current head and kept the stack moving.
- No upstream-side mutation was performed; all changes stay on the `easel` side of the mirror.
