# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T20:04:51-04:00
Current base: `origin/main` `8782d07a`
Current integration tip before this refresh: `easel/auto-integration` `74f6e5bd`
Refreshed stack merge commit prepared in this run: none; stack already current
Final manifest commit prepared after stack merge: this commit

This branch is maintained as a reproducible patch stack over `origin/main`.
The upstream base and carried PR heads were already current at the start of this
unattended run, so the work focused on revalidating open non-draft PR
classification, probing still-blocked merges in isolated worktrees, and recording
that no stack rewrite was needed.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #297 | `feat-server-laguna-layer-split-adapter-v2` | `53dd1686` | merged in this run | Draft but green/mergeable Laguna target-layer-split adapter; now carried in the easel stack with shared layer-split plumbing and Laguna target/runtime files. |
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
| #285 | `feat/lucebox-docker` | draft | partial draft dependency | Draft PR, outside the primary non-draft target, but earlier Docker/CLI/bench integration dependency remains carried. |

## Fresh probe results from this run

| PR | Outcome | Notes |
|---:|---|---|
| current upstream/base | checked | `git merge --no-edit origin/main` in an isolated worktree reported `Already up to date.`; no upstream reconciliation commit was needed. |
| #297 | merged | `git merge --no-commit --no-ff pr-297` completed cleanly in the isolated worktree, then the merge was committed on `auto-integration`. |
| current integrated PRs | checked | Ancestor checks still pass for the open non-draft contributor PR refs already carried in the stack: #295, #294, #289, #276, #274, #266, #152, and #142. |
| upstream-adjacent PRs | checked | #292/#298/#299/#300/#301/#302/#303 are included through `origin/main`. |
| #237 | blocked-needs-human / selective-port | Fresh direct merge probe still conflicts across deleted legacy `dflash/` server files, `server/CMakeLists.txt`, backend factory/model backend, MTP common interfaces/orchestrator, Qwen35 loader/graph/backend/target files, and tests. A tmux-driven Claude attempt ended at max turns without a usable report; a tmux-driven Codex feasibility review concluded this is not a safe mechanical merge because MTP API/loader/graph/decode behavior collides with current remote-draft, layer-split, pFlash, thinking-budget, telemetry, and MoE hooks. |
| #221 | blocked-needs-human / dependency | Fresh direct merge probe still conflicts across old prefix-cache/MTP/common/Qwen35 files, server tests, legacy script paths, and benchmark artifacts. It still depends on a current-layout #237-equivalent MTP foundation before a useful port can be made. |
| #154 | blocked-needs-human / dependency | Fresh direct merge probe conflicts in legacy `dflash/CMakeLists.txt`, MTP docs, `server/src/internal.h`, Qwen35 loader/target graph, and MTP smoke/contract tests. Portable only after current-layout Qwen35 MTP exists. |
| #153 | blocked-needs-human / dependency | Fresh direct merge probe shows the same old-layout MTP integrated runtime conflict class as #154; requires current-layout loader/graph/cache design work. |
| #137 | suggested-close/superseded | Fresh direct merge probe conflicts only in legacy `dflash/CMakeLists.txt`; current `server/CMakeLists.txt` owns CUDA arch/BSA handling. |
| #135 | blocked-needs-human / selective-port | Fresh direct merge probe conflicts in `server/src/internal.h`, `server/src/qwen35/qwen35_target_graph.cpp`, and `server/test/test_dflash.cpp`. Needs current multi-request scheduler design rather than direct conflict resolution. |
| #94 | suggested-close/superseded | Fresh direct merge probe conflicts in current draft graph/safetensors loader/internal files. Current code already carries the useful SWA draft parsing and causal-mask behavior; remaining branch edits are old-layout/obsolete. |
| #48 | suggested-close/superseded | Fresh direct merge probe conflicts only in legacy `dflash/CMakeLists.txt`; current server CMake supersedes the old Blackwell arch detection patch. |

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
ongoing dependency awareness: #291, #290, #286, #285, #275, #249, and #193.
#286 is the draft PR for the current auto-integration snapshot.

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T20:04:51-04:00 for the manifest refresh timestamp.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status`, `HOME=/home/erik claude auth status --text`, and `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version` all succeeded.
- `git fetch --prune origin` and `git fetch --prune easel` completed separately; targeted pull-ref fetch recreated open PR refs.
- Isolated worktree `/tmp/luce-auto-cron-20260528-200554` was created from `easel/auto-integration`; `git merge --no-edit origin/main` reported `Already up to date.`
- Ancestor checks still pass for carried open non-draft PR refs: #295, #294, #289, #276, #274, #266, #152, and #142. Draft #297 remains carried at `53dd1686`.
- Fresh direct merge probes were run in isolated worktrees for the still-unintegrated non-draft PRs #237, #221, #154, #153, #137, #135, #94, and #48; each still conflicts as summarized above. Probe log: `/tmp/luce-probe-results-20260528-200603.txt`.
- A tmux-driven Claude report attempt for #237 ended at `Error: Reached max turns (8)` without usable output; the failed report is `/tmp/luce-pr237-claude-20260528-200614.txt`.
- A tmux-driven Codex inspection for #237 completed with no edits and recommended keeping #237 blocked for manual design. Report: `/tmp/luce-pr237-codex-20260528-200706.txt`.
- Manifest-only validation on the reconciliation worktree: `git diff --check HEAD -- docs/auto-integration.md` passed and `git diff --check origin/main...HEAD` still reports only the known pre-existing whitespace warnings outside this run's change (`luce-bench/src/lucebench/fixtures/forge_eval/scenarios/_model_quality.py`, `_stateful_model_quality.py`, and `scripts/docker_build_env.sh`).
- Anchored conflict-marker scan on the reconciliation worktree `git grep -n -E '^(<<<<<<<|>>>>>>>|=======)( |$)' -- . ':!server/eval/humaneval_plus/humanevalplus.jsonl'` found no markers.

## Notes

- This run produced a manifest-only refresh on top of `74f6e5bd`; no source stack rewrite was needed because `origin/main` and all carried PR heads were unchanged.
- Retained reconciliation worktree `/tmp/luce-auto-cron-20260528-200554` contains this manifest refresh.
- Retained fresh conflict probe worktrees `/tmp/luce-probe-pr237-20260528-200603`, `/tmp/luce-probe-pr221-20260528-200603`, `/tmp/luce-probe-pr154-20260528-200603`, `/tmp/luce-probe-pr153-20260528-200603`, `/tmp/luce-probe-pr137-20260528-200603`, `/tmp/luce-probe-pr135-20260528-200603`, `/tmp/luce-probe-pr94-20260528-200603`, and `/tmp/luce-probe-pr48-20260528-200603` are intentionally left for review because they are conflicted.
- Prior retained worktrees, probe logs, agent reports, and configure directories remain as listed in earlier manifest revisions; cleanup is separate maintenance.
