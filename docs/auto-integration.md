# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T11:03:15-04:00
Current base: `origin/main` `43457d8f`
Current integration tip before this refresh: `easel/auto-integration` `f3dfad87`
Refreshed stack tip prepared in this run: this commit

This branch is maintained as a reproducible patch stack over `origin/main`.
The primary checkout was clean at the start of this unattended run. This refresh
keeps the stack rebased on the unchanged upstream base, integrates the latest
non-draft contributor update from PR #289, reruns direct conflict probes for the
remaining non-ancestor non-draft PRs, and records one additional delegated
read-only feasibility attempt for PR #221.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #292 | `feat-backend-ipc-payload-pipe-open` | `90bc52f` | included | Adds backend IPC payload-pipe support for remote draft feature/noise payloads, with integration-only pipe-drain hardening and shared feature-slice storage helper. |
| #289 | `pipeline_moe` | `d9917c7` | included / refreshed this run | Latest pipelined hybrid Qwen35 MoE decode update merged cleanly. This refresh adds selected expert-weight normalization, guards the expert-weight scale default, and removes a noisy pipelined decode init printf. The inaccessible submodule pointer from earlier PR history remains excluded. |
| #284 | `fix/draft-safetensors-rope-theta` | `63bba30` | included | Reads and validates `rope_theta` from draft safetensors `config.json`. |
| #278 | `fix-pflash-drafter-backend-precision-submit` | `fdfcbda` | included | Adds legacy CUDA drafter precision fallback via shared backend precision policy while preserving current Q8_0 allocation behavior where present. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `0e3c79a` | included | Qwen3.6-27B tool-calling fix for Claude-code Anthropic path. |
| #274 | `feat/pflash-drafter-ee7` | `9c9aee9` | included | Adaptive pFlash compression docs/config, transitive anchor default, backup-suffix ignores, EE7/drafter updates, and related server changes are carried. |
| #273 | `feat-cpp-server-gemma4-layer-split-adapter` | `79abba9` | included | Gemma4 target-layer-split adapter and loader/graph support. |
| #266 | `feat/harness-typed-adapters` | `17525ea` | included | Typed harness adapters and format-aware session-inject proxy. |
| #265 | `feat-cpp-server-target-layer-split-prep` | `054af28` | included through upstream | Upstream main contains this PR. |
| #177 | `split/gemma4-06-kv-correctness` | `0a95d4b` | selectively included | Previously carried split-on-wrap SWA KV writes and larger SWA ring allocation for chunked long-context prefill in `server/src/gemma4/gemma4_loader.cpp`. Remaining TQ3/large-head KV alignment and tests still require manual design. |
| #174 | `split/gemma4-14-small-vram-docs` | `8b1caba` | selectively included | Useful small-VRAM/VMM documentation is already ported into current `server/README.md`. |
| #152 | `main` | `cf735be` | included | Gemma 4 RTX 4090 backend helpers already carried. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | `f2fbf62` | included | FP16 safetensors drafter support already carried. |
| #94 | `feat/dflash-qwen36-swa-draft` | `d2f9c9d` | absorbed / selectively included | Current draft/common code carries safetensors SWA config parsing and causal-mask behavior; remaining branch edits target old paths. |
| #62 | `fix/issue-55-stable-kv-pad` | `0ce6832` | absorbed / selectively included | Current server layout carries daemon reset behavior and regression coverage; remaining branch conflicts are legacy tests. |
| #48 | `fix/consumer-blackwell-auto-detect` | `858b84b` | superseded / selectively included | Current `server/CMakeLists.txt` owns CUDA architecture resolution and Blackwell/GB10 handling. |
| #39 | `feat/moe-35b-a3b` | `c86ec86` | partially integrated / mostly superseded | Draft safetensors `config.json`/YaRN survivorship was previously ported; remaining old loader/FFN/target-graph pieces are superseded by current qwen35moe/DDTree code. |
| #285 | `feat/lucebox-docker` | draft | partial draft dependency | Draft PR, outside the non-draft target, but earlier Docker/CLI/bench integration dependency remains carried. |

## Fresh probe results from this run

| PR | Outcome | Notes |
|---:|---|---|
| upstream sync | checked | `origin/main` remained `43457d8f`; reconciliation worktree `/tmp/luce-auto-cron-20260528-105846` started from `easel/auto-integration` `f3dfad87` and was already up to date with upstream. |
| #289 | integrated | `origin/pr/289` advanced from the previously included `4933ce7` to `d9917c7`. A direct merge into the reconciliation worktree completed without conflicts and was committed as a separate stack commit before this manifest refresh. |
| current integrated PRs | checked | After the #289 refresh, `git merge-base --is-ancestor origin/pr/<n> HEAD` shows #292, #289, #284, #278, #276, #274, #273, #266, #152, and #142 are ancestors of the refreshed stack. #265 is included through `origin/main`. |
| remaining non-ancestor non-draft PRs | direct merge probes still conflicted | Fresh isolated probes attempted `--no-commit --no-ff` merges for #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, and #39 against the post-#289 stack. Every direct probe conflicted and was aborted in the isolated probe worktree. Consolidated output: `/tmp/luce-merge-probes-20260528-105846.txt`. |
| #221 delegated read-only review | attempted; inconclusive | Codex was launched through tmux with a read-only feasibility prompt. It produced a large inspection transcript but did not converge to the requested concise report before the session was stopped; retained output: `/tmp/luce-codex-pr221-20260528-105846.txt`. Existing classification remains dependency-blocked on a current-layout MTP foundation equivalent to #237. |

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Current status | Next useful action |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Direct merge still conflicts across current server/common/qwen35 files and old `dflash/` paths. Prior Codex review recommended a deliberate current-layout MTP architecture port, not conflict-marker resolution. Minimal candidates: `server/src/common/mtp_*`, Qwen35 MTP graph/loader files, backend optional MTP hooks, native-server MTP CLI flags, CMake wiring, and adapted orchestrator/tests. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Requires #237/equivalent MTP foundation, then a current-layout feature port for hidden-state/speculator hooks, backend dispatcher, daemon wiring, and WARM tests. This run's Codex attempt did not produce a usable narrower plan. |
| #183 | `split/gemma4-11a-target-mtp-integration` | blocked-needs-human / selective-port | Direct merge is unsafe because it targets retired `dflash/` + flat Gemma4 paths. Port KV fix, `h_prev`/asymmetric KV hook, MTP loader/graph, hardening, and tests into current `server/src/gemma4/*`. |
| #182 | `split/gemma4-10-mtp-loader-step-graph` | blocked-needs-human / selective-port | Port MTP declarations/cache fields, assistant loader pieces, a current `gemma4_mtp_graph.cpp`, CMake wiring, and tests into current layout. |
| #181 | `split/gemma4-09-dflash-draft-runtime` | blocked-needs-human / selective-port | Mine quantizer metadata/error-reporting/runtime metadata concepts and reauthor tests against current Gemma4/draft plumbing. |
| #180 | `split/gemma4-08-draft-loader-quant` | blocked-needs-human / selective-port | Reconcile quantizer metadata with current `draft_gguf_loader.cpp`, validate capture/target-layer IDs, and reauthor smoke tests. |
| #177 | `split/gemma4-06-kv-correctness` | partially integrated / selective-port | SWA split-on-wrap and larger SWA ring allocation are ported. Remaining work: TQ3/large-head KV alignment/type resolution, FA rotation path, and adapted `test_gemma4_kv_tq3.cpp`. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | selective-port | Linear native MTP decode semantics are portable after a current-layout Qwen35 MTP design. |
| #153 | `xabicasa/dflash-mtp-integrated` | blocked-needs-human / selective-port | Feasible but requires loader/graph/cache/MoE design work, not conflict-marker resolution. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | stale / suggested close | Edits only deleted legacy `dflash/CMakeLists.txt`; close or ask author to retarget current `server/CMakeLists.txt`. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | blocked-needs-human / selective-port | Port `n_seqs` primitives/test helpers first, then design multi-slot `Qwen35Backend` state and scheduler API. |
| #131 | `feature/gemma4-support` | selective-port | Broad Gemma4 support can be mined, but direct merge would resurrect old `dflash27b`/`dflash/` layout. |
| #94 | `feat/dflash-qwen36-swa-draft` | absorbed / verify-only | Remaining conflicts are old-layout draft/common paths; current code carries the useful SWA pieces. |
| #62 | `fix/issue-55-stable-kv-pad` | absorbed / verify-only | Remaining conflicts are old-layout tests; current code carries reset behavior. |
| #48 | `fix/consumer-blackwell-auto-detect` | superseded / suggested close | Current CMake supersedes the old `dflash/CMakeLists.txt` change. |
| #39 | `feat/moe-35b-a3b` | partially integrated / mostly superseded | Ask maintainers whether any remaining old MoE smoke coverage should be reauthored against current qwen35moe APIs before closing. |

## Draft / excluded

Draft PRs remain outside the primary non-draft integration target except for
dependency awareness: #291, #290, #286, #285, #275, #249, #193, and #75. #286 is
the current draft auto-integration snapshot. #285 remains partially carried only
as an integration dependency.

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T10:57:45-04:00 at preflight and 2026-05-28T11:03:15-04:00 at manifest refresh.
- Primary checkout `git status --short` was clean before work began.
- `git branch --show-current` reported `auto-integration` in the primary checkout.
- `git remote -v` verified `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status` succeeded for account `easel` with repo/workflow scopes.
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text` succeeded for the Claude Team account.
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version` succeeded and reported `codex-cli 0.130.0`.
- `git fetch --prune origin` and `git fetch --prune easel` completed; targeted fetches recreated current open non-draft contributor PR refs.
- Isolated reconciliation worktree `/tmp/luce-auto-cron-20260528-105846` merged latest `origin/main` and was already up to date.
- `origin/pr/289` merged cleanly into the reconciliation worktree and was committed as `bb1738d` before this manifest refresh.
- Isolated probe worktree `/tmp/luce-probe-after289-20260528-105846` reran direct conflict probes for all remaining non-ancestor non-draft PRs.
- Tmux-driven Codex read-only PR #221 review was attempted but produced only a large inspection transcript, not the requested concise plan.
- `git diff --cached --check` passed for the #289 refresh commit before commit.
- `git diff --check HEAD~2..HEAD` passed for the final two-commit refresh stack.
- Targeted conflict-marker scan over files changed by `HEAD~2..HEAD` found no merge markers.
- Ancestor checks passed for included contributor PR refs #292, #289, #284, #278, #276, #274, #273, #266, #152, and #142.
- `cmake -S server -B /tmp/luce-cmake-20260528-105846 -DDFLASH27B_GPU_BACKEND=cuda -DCMAKE_CUDA_ARCHITECTURES=89` failed during CUDA compiler identification because local `/usr/bin/nvcc`/CMake still selected unsupported `sm_52` (`ptxas fatal : Value 'sm_52' is not defined`), before project compilation.
- Push outcome is recorded in the final cron report.

## Notes

- Primary checkout `/home/erik/Projects/luce2` was clean at preflight and matched fetched `easel/auto-integration` (`f3dfad87`).
- Retained reconciliation worktree `/tmp/luce-auto-cron-20260528-105846` for audit/final push preparation.
- Retained probe worktrees `/tmp/luce-probe-20260528-105846` and `/tmp/luce-probe-after289-20260528-105846` plus direct-probe log `/tmp/luce-merge-probes-20260528-105846.txt`.
- Retained Codex output `/tmp/luce-codex-pr221-20260528-105846.txt`.
- Prior retained worktrees, probe logs, agent reports, and configure directories remain as listed in earlier manifest revisions; cleanup is separate maintenance.
