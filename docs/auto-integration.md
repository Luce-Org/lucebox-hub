# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T19:11:53-04:00
Current base: `origin/main` `8782d07a`
Current integration tip before this refresh: `easel/auto-integration` `dbf88ac3`
Refreshed stack merge commit prepared in this run: no stack merge required
Final manifest commit prepared after stack merge: this commit

This branch is maintained as a reproducible patch stack over `origin/main`.
The primary checkout was clean at the start of this unattended run. `origin/main`
and `easel/auto-integration` were already reconciled: the integration branch
has `origin/main` as its merge-base and no new upstream commits were pending.
This refresh revalidated the open non-draft contributor PR set after fetching
current pull refs and confirmed that all direct-mergeable non-draft contributor
PRs remain ancestors of the stack. Remaining non-ancestor PRs still conflict in
old-layout MTP/scheduler or superseded CMake/draft areas and require selective
current-layout ports rather than direct merges.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #303 | `fix/harness-portable-run-dirs` | upstream `05b008a0` | included through upstream and stack | Harness portable run/cache directories and automatic client-install fallback are in `origin/main`; the stack merge preserved local compatibility docs and helpers. |
| #302 | `fix/harness-model-paths` | upstream | included through upstream | Harness launcher model-path override behavior and documentation are in `origin/main`. |
| #301 | `fix/ddtree-test-harness` | upstream | included through upstream | DDTree test harness fixes are in `origin/main`. |
| #300 | `fix/sigterm-gpu-unload` | upstream | included through upstream | SIGTERM GPU-unload fix remains in upstream. |
| #299 | `feat/draft-swa-flag` | upstream | included through upstream | Draft SWA env/flag support remains in upstream. |
| #298 | `fix/gemma4-destructor-link` | upstream | included through upstream | Gemma4 destructor-link fix remains in upstream. |
| #292 | `feat-backend-ipc-payload-pipe-open` | upstream / `90bc52f` | included through upstream and stack | `origin/main` carries backend IPC payload-pipe support; the stack already contained the same PR head before it landed upstream. |
| #295 | `fix-layer-split-sampling` | `a9aedf7d` | included | Target layer-split sampling support remains an ancestor of the stack. |
| #294 | `feat/server-passthrough-proxy` | `0883c2e` | included | Server passthrough proxy wiring, piecewise keep-ratio curve, query survival checks, and unit coverage are carried. |
| #289 | `pipeline_moe` | `0ffab8a` | included | Pipelined hybrid Qwen35 MoE decode update is carried; inaccessible submodule history remains excluded. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `5e861b4` | included | Qwen3.6-27B tool-calling fix for Claude-code Anthropic path is carried. |
| #274 | `feat/pflash-drafter-ee7` | `e64a2b8` | included | Adaptive pFlash composition, EE7/drafter updates, docs, tests, and follow-up fixes are carried. |
| #266 | `feat/harness-typed-adapters` | `17525ea` | included | Typed harness adapters and format-aware session-inject proxy are carried. |
| #152 | `main` | `cf735be` | included | Gemma 4 RTX 4090 backend helpers are carried. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | `f2fbf62` | included | FP16 safetensors drafter support is carried. |
| #285 | `feat/lucebox-docker` | draft | partial draft dependency | Draft PR, outside the non-draft target, but earlier Docker/CLI/bench integration dependency remains carried. |

## Fresh probe results from this run

| PR | Outcome | Notes |
|---:|---|---|
| current upstream/base | checked | `git merge --no-edit origin/main` in the isolated refresh worktree reported `Already up to date.`; no upstream reconciliation commit was needed. |
| current integrated PRs | checked | `git merge-base --is-ancestor origin/pr/<n> HEAD` passed for open non-draft PRs #295, #294, #289, #276, #274, #266, #152, and #142. #292/#298/#299/#300/#301/#302/#303 are included through `origin/main`. |
| #237 | blocked-needs-human / selective-port | Direct merge still conflicts across legacy `dflash/` server paths, `server/CMakeLists.txt`, backend factory/model backend, MTP common interfaces/orchestrator, Qwen35 loader/graph/backend/target files, and tests. Claude tmux delegation reached `--max-turns` without a usable report; Codex tmux delegation completed read-only and confirmed selective-port, not direct integration: the PR is ~7.2k added lines against the old `dflash/src` layout, current code lacks `server/src/common/mtp_interface.h`, `mtp_chain_runner.*`, `mtp_orchestrator.*`, `gguf_metadata.h`, and `server/src/qwen35/qwen35_mtp*`, and the next smallest safe step is a current-layout no-op MTP API/orchestrator-test port before Qwen35 runtime wiring. |
| #221 | blocked-needs-human / dependency | Direct merge still conflicts across old prefix-cache/MTP/common/Qwen35 files, server tests, and legacy script paths. It depends on a current-layout #237-equivalent MTP foundation before a useful port can be made. |
| #154 | blocked-needs-human / dependency | Direct merge conflicts in legacy `dflash/CMakeLists.txt`, MTP docs, `server/src/internal.h`, Qwen35 loader/target graph, and MTP smoke/contract tests. Portable only after current-layout Qwen35 MTP exists. |
| #153 | blocked-needs-human / dependency | Same old-layout MTP integrated runtime conflict class as #154; requires current-layout loader/graph/cache design work. |
| #137 | suggested-close/superseded | Direct merge conflicts only in legacy `dflash/CMakeLists.txt`; current `server/CMakeLists.txt` owns CUDA arch/BSA handling. |
| #135 | blocked-needs-human / selective-port | Direct merge conflicts in `server/src/internal.h`, `server/src/qwen35/qwen35_target_graph.cpp`, and `server/test/test_dflash.cpp`. Needs current multi-request scheduler design rather than direct conflict resolution. |
| #94 | suggested-close/superseded | Direct merge conflicts in current draft graph/safetensors loader/internal files. Current code already carries the useful SWA draft parsing and causal-mask behavior; remaining branch edits are old-layout/obsolete. |
| #48 | suggested-close/superseded | Direct merge conflicts only in legacy `dflash/CMakeLists.txt`; current server CMake supersedes the old Blackwell arch detection patch. |

Direct-probe log: `/tmp/luce-merge-probes-20260528-191128.txt`.
Worktree used for this refresh: `/tmp/luce-auto-cron-20260528-191128`.
Claude tmux delegation report for #237 from prior substantial selective-port attempt: `/tmp/luce-claude-pr237-20260528-182754.txt` (unusable: max turns).
Codex tmux delegation report for #237 from prior substantial selective-port attempt: `/tmp/luce-codex-pr237-20260528-182754.txt`.

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Current status | Next useful action |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | First port only the common MTP API surface to `server/src/common`: no-op `DFlashTarget`/`ModelBackend` hooks, `mtp_interface`, chain runner/orchestrator, CMake entries, and `test_common_mtp_orchestrator`; after that compiles, port Qwen35 MTP graph/loader/runtime while preserving current remote-draft, pFlash, layer-split, thinking-budget, telemetry, and MoE hooks. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Revisit after #237-equivalent MTP foundation; then port hidden-state/speculator hooks, backend dispatcher, daemon wiring, and WARM tests to current `server/` layout. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | blocked-needs-human / dependency | Mine linear MTP decode semantics after current-layout Qwen35 MTP exists. |
| #153 | `xabicasa/dflash-mtp-integrated` | blocked-needs-human / dependency | Mine loader/graph/cache/test ideas after current-layout Qwen35 MTP exists. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | blocked-needs-human / selective-port | Port only after designing current multi-slot `Qwen35Backend` state and scheduler API. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | suggested-close/superseded | Ask author to close or retarget to current `server/CMakeLists.txt` if anything remains. |
| #94 | `feat/dflash-qwen36-swa-draft` | suggested-close/superseded | Useful behavior appears absorbed; ask author/maintainers whether any remaining old-layout tests should be reauthored before close. |
| #48 | `fix/consumer-blackwell-auto-detect` | suggested-close/superseded | Close or retarget to current `server/CMakeLists.txt` if still needed. |

## Draft / excluded

Draft PRs remain outside the primary non-draft integration target except for
dependency awareness: #297, #291, #290, #286, #285, #275, #249, and #193.
#286 is the draft PR for the current auto-integration snapshot.

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T19:10:43-04:00 for the preflight timestamp and 2026-05-28T19:11:53-04:00 for the manifest refresh timestamp.
- Primary checkout `/home/erik/Projects/luce2` was clean before starting.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status` passed for account `easel`; `HOME=/home/erik claude auth status --text` passed; `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version` returned `codex-cli 0.130.0`.
- `git fetch --prune origin` and `git fetch --prune easel` completed separately; targeted pull-ref fetch recreated PR refs (with a non-fatal inaccessible historical BSA submodule warning).
- Worktree `/tmp/luce-auto-cron-20260528-191128` was created from `easel/auto-integration` `dbf88ac3` for isolated integration and probes.
- `git merge --no-edit origin/main` reported `Already up to date.` in the worktree.
- Isolated direct probes attempted `git merge --no-commit --no-ff origin/pr/<n>` for #237, #221, #154, #153, #137, #135, #94, and #48; each conflicted and was aborted in the worktree. Log: `/tmp/luce-merge-probes-20260528-191128.txt`.
- Ancestor checks passed for included open non-draft contributor PR refs #295, #294, #289, #276, #274, #266, #152, and #142.
- Prior #237 delegated attempts remain the latest substantial agent work: tmux sessions `luce237-182754` (Claude, max-turns without usable report) and `luce237codex-182754` (Codex read-only report completed).
- `bash -n harness/clients/common.sh` passed.
- `git diff --check origin/main...HEAD` still reports three pre-existing whitespace warnings outside this run's changes (`luce-bench/src/lucebench/fixtures/forge_eval/scenarios/_model_quality.py`, `_stateful_model_quality.py`, and `scripts/docker_build_env.sh`); `git diff --check dbf88ac3...HEAD` passed for this run's manifest delta.
- Anchored conflict-marker scan `git grep -n -E '^(<<<<<<<|>>>>>>>|=======)( |$)' -- . ':!server/eval/humaneval_plus/humanevalplus.jsonl'` found no markers.

## Notes

- Retained worktree `/tmp/luce-auto-cron-20260528-191128` and direct-probe log `/tmp/luce-merge-probes-20260528-191128.txt`.
- Prior retained worktrees, probe logs, agent reports, and configure directories remain as listed in earlier manifest revisions; cleanup is separate maintenance.
