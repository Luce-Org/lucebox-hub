# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T09:01:27-04:00
Current base: `origin/main` `43457d8`
Current integration tip before this refresh: `easel/auto-integration` `e57782c`
Refreshed stack tip prepared in this run: `9a7d642`

This branch is maintained as a reproducible patch stack over `origin/main`.
The primary checkout was clean at the start of this unattended run. This refresh
merged the latest upstream `origin/main` and integrated the new non-draft
contributor PR #292. The PR #292 merge conflicted only in the draft IPC daemon;
the conflict was resolved by preserving the existing feature-range commands while
adding the new payload-pipe commands. A Codex review caught a pipe-drain edge
case in the first resolution, and this run fixed it before verification.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #292 | `feat-backend-ipc-payload-pipe-open` | `2ad882c` | included this run | Adds backend IPC payload-pipe support for remote draft feature/noise payloads. Manual conflict resolution in `server/src/common/dflash_draft_ipc_daemon.cpp` preserved `feature_slice`, `get_feature_range`, `set_feature_range`, and `propose` while adding `feature_slice_pipe`/`propose_pipe`; invalid pipe commands now drain the advertised payload before returning failure to avoid writer stalls or stale pipe bytes. |
| #289 | `pipeline_moe` | `4933ce7` | included | Adds pipelined hybrid Qwen35 MoE decode with persistent decode state and optimized FFN routing. The inaccessible submodule pointer from the PR was not adopted. |
| #284 | `fix/draft-safetensors-rope-theta` | `63bba30` | included | Reads and validates `rope_theta` from draft safetensors `config.json`. |
| #278 | `fix-pflash-drafter-backend-precision-submit` | `fdfcbda` | included | Adds legacy CUDA drafter precision fallback via shared backend precision policy while preserving current Q8_0 allocation behavior where present. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `0e3c79a` | included | Qwen3.6-27B tool-calling fix for Claude-code Anthropic path. |
| #274 | `feat/pflash-drafter-ee7` | `5037b28` | included | EE7 early-exit drafter support. |
| #273 | `feat-cpp-server-gemma4-layer-split-adapter` | `79abba9` | included | Gemma4 target-layer-split adapter and loader/graph support. |
| #266 | `feat/harness-typed-adapters` | `17525ea` | included | Typed harness adapters and format-aware session-inject proxy. |
| #265 | `feat-cpp-server-target-layer-split-prep` | `054af28` | included | Upstream main now contains this PR; included through the upstream sync. |
| #152 | `main` | `cf735be` | included | Gemma 4 RTX 4090 backend helpers already carried. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | `f2fbf62` | included | FP16 safetensors drafter support already carried. |
| #177 | `split/gemma4-06-kv-correctness` | `0a95d4b` | selectively included | SWA ring-buffer wrap-safe K/V cache write was previously ported into current `server/src/gemma4/gemma4_graph.cpp`; remaining old-layout loader/graph/test files still require deliberate mapping. |
| #174 | `split/gemma4-14-small-vram-docs` | `8b1caba` | selectively included | Useful small-VRAM/VMM documentation is already ported into current `server/README.md`. |
| #94 | `feat/dflash-qwen36-swa-draft` | `d2f9c9d` | absorbed / selectively included | Current draft/common code carries safetensors SWA config parsing and causal-mask behavior; remaining branch edits target old paths. |
| #62 | `fix/issue-55-stable-kv-pad` | `0ce6832` | absorbed / selectively included | Current server layout carries daemon reset behavior and regression coverage; remaining branch conflicts are legacy tests. |
| #48 | `fix/consumer-blackwell-auto-detect` | `858b84b` | superseded / selectively included | Current `server/CMakeLists.txt` owns CUDA architecture resolution and Blackwell/GB10 handling. |
| #39 | `feat/moe-35b-a3b` | `c86ec86` | partially integrated / mostly superseded | Draft safetensors `config.json`/YaRN survivorship was previously ported; remaining old loader/FFN/target-graph pieces are superseded by current qwen35moe/DDTree code. |
| #285 | `feat/lucebox-docker` | draft | partial draft dependency | Draft PR, outside the non-draft target, but earlier Docker/CLI/bench integration dependency remains carried. |

## Fresh probe results from this run

| PR | Outcome | Notes |
|---:|---|---|
| upstream sync | merged | Isolated worktree `/tmp/luce-auto-cron-20260528-085422` merged `origin/main` `43457d8` into prior `easel/auto-integration` `e57782c`, producing merge commit `869f8e7`. |
| #292 | integrated | Direct merge conflicted in `server/src/common/dflash_draft_ipc_daemon.cpp`; resolved manually and committed as merge `9a7d642`. Claude-in-tmux review `/tmp/pr292-claude-review-20260528-085422.txt` hit `--max-turns`; Codex-in-tmux review `/tmp/pr292-codex-review-20260528-085422.txt` found one pipe-drain issue, fixed before this manifest update. |
| current integrated PRs | checked | `git merge-base --is-ancestor origin/pr/<n> HEAD` shows #292, #289, #284, #278, #276, #274, #273, #266, #152, and #142 are ancestors of the refreshed stack. #265 is now included through `origin/main`. |
| remaining non-ancestor non-draft PRs | direct merge probes still conflicted | Fresh isolated probes attempted `--no-commit --no-ff` merges for #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39. Every direct probe conflicted and was aborted in the isolated worktree. Consolidated output: `/tmp/luce-merge-probes-20260528-085422.txt`. |

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Current status | Next useful action |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Direct merge still conflicts. Prior Codex/Claude reports recommend a deliberate full MTP architecture port rather than cherry-picking server-only pieces. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Requires #237/equivalent MTP foundation, then a current-layout feature port for hidden-state/speculator hooks, backend dispatcher, daemon wiring, and WARM tests. |
| #183 | `split/gemma4-11a-target-mtp-integration` | blocked-needs-human / selective-port | Direct merge is unsafe because it targets retired `dflash/` + flat Gemma4 paths. Port KV fix, `h_prev`/asymmetric KV hook, MTP loader/graph, hardening, and tests into current `server/src/gemma4/*`. |
| #182 | `split/gemma4-10-mtp-loader-step-graph` | blocked-needs-human / selective-port | Port MTP declarations/cache fields, assistant loader pieces, a current `gemma4_mtp_graph.cpp`, CMake wiring, and tests into current layout. |
| #181 | `split/gemma4-09-dflash-draft-runtime` | blocked-needs-human / selective-port | Mine quantizer metadata/error-reporting/runtime metadata concepts and reauthor tests against current Gemma4/draft plumbing. |
| #180 | `split/gemma4-08-draft-loader-quant` | blocked-needs-human / selective-port | Reconcile quantizer metadata with current `draft_gguf_loader.cpp`, validate capture/target-layer IDs, and reauthor smoke tests. |
| #177 | `split/gemma4-06-kv-correctness` | partially integrated / selective-port | Port only net-new `gemma4_last_error()` thread-local handling, target structs/loader/graph deltas, CUDA forward declaration cleanup, and smoke/KV tests if still valuable. |
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

- `date -Is` -> 2026-05-28T08:53:34-04:00 at preflight and 2026-05-28T09:01:27-04:00 at manifest refresh.
- Primary checkout `git status --short` was clean before work began.
- `git remote -v` verified `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status` succeeded for account `easel`.
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text` succeeded for the Claude Team account.
- Harmless `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version` smoke check succeeded (`codex-cli 0.130.0`).
- `git fetch --prune origin` and `git fetch --prune easel` completed; targeted fetches recreated current open non-draft PR refs.
- `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 200 --json ... --jq ...` enumerated all open PRs and showed #291, #290, #286, #285, #275, #249, #193, and #75 as drafts/excluded.
- Isolated reconciliation worktree `/tmp/luce-auto-cron-20260528-085422` merged `origin/main` and PR #292, then reran direct conflict probes for the remaining non-ancestor non-draft PRs.
- Claude-in-tmux review for #292 reached the turn limit with no usable report; Codex-in-tmux review completed, found the payload-pipe drain issue, and the issue was fixed.
- `git diff --check HEAD~1..HEAD` passed after the #292 merge/fix.
- Full-stack `git diff --check origin/main...HEAD` still reports pre-existing trailing blank line warnings in `luce-bench/src/lucebench/fixtures/forge_eval/scenarios/_model_quality.py` and `_stateful_model_quality.py`.
- Search for exact merge conflict markers (`^(<<<<<<<|=======|>>>>>>>)`) under the reconciliation worktree returned no results.
- `cmake -S server -B /tmp/luce-build-20260528-085422 -DLUCE_BUILD_TESTS=ON` failed during CUDA compiler identification before project compilation with the known local nvcc/CMake `sm_52` toolchain issue (`ptxas fatal : Value 'sm_52' is not defined for option 'gpu-name'`).

## Notes

- Primary checkout `/home/erik/Projects/luce2` was clean at preflight and matched fetched `easel/auto-integration` (`e57782c`).
- Retained worktree `/tmp/luce-auto-cron-20260528-085422` for audit/final push preparation.
- Retained direct-probe log `/tmp/luce-merge-probes-20260528-085422.txt`.
- Retained Claude/Codex review reports `/tmp/pr292-claude-review-20260528-085422.txt` and `/tmp/pr292-codex-review-20260528-085422.txt`.
- Retained configure directory `/tmp/luce-build-20260528-085422` showing the local CUDA compiler-identification blocker.
- Prior retained worktrees, probe logs, and agent reports remain as listed in earlier manifest revisions; cleanup is separate maintenance.
