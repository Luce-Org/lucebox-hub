# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T16:29:05-04:00
Current base: `origin/main` `3ba525e0`
Current integration tip before this refresh: `easel/auto-integration` `ea777e0c`
Refreshed stack tip prepared in this run: this commit

This branch is maintained as a reproducible patch stack over `origin/main`.
The primary checkout was clean at the start of this unattended run. `origin/main`
was already an ancestor of `easel/auto-integration`, so no upstream merge was
needed. This run refreshed the manifest after re-enumerating open PRs, fetching
all open non-draft contributor refs, verifying the included PR heads, and rerunning
isolated direct-merge probes for the remaining non-ancestor PR heads.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #298 | `fix/gemma4-destructor-link` | `32ba96e9` | included | Gemma4 destructor-link fix is an ancestor of the stack. |
| #295 | `fix-layer-split-sampling` | `a9aedf7d` | included | Target layer-split sampling support is an ancestor of the stack. |
| #294 | `feat/server-passthrough-proxy` | `0883c2e` | included | Server passthrough proxy wiring, piecewise keep-ratio curve, query survival checks, and unit coverage are carried. |
| #292 | `feat-backend-ipc-payload-pipe-open` | `90bc52f` | included | Backend IPC payload-pipe support for remote draft feature/noise payloads is carried. |
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
| upstream sync | checked | `origin/main` `3ba525e0` is already an ancestor of `easel/auto-integration` `ea777e0c`; no upstream merge was required. |
| current integrated PRs | checked | `git merge-base --is-ancestor origin/pr/<n> HEAD` passed for open non-draft PRs #298, #295, #294, #292, #289, #276, #274, #266, #152, and #142. |
| #237 | blocked-needs-human / selective-port | Direct merge still conflicts across moved legacy `dflash/` paths, common MTP interfaces, Qwen35 backend/graph/loader files, CMake, daemon/server wiring, and tests. A tmux-driven Codex feasibility pass (`/tmp/luce-pr237-codex-20260528-162440.txt`) concluded there is no safe mechanical production port; the next step is a deliberate current-layout MTP design/port. |
| #221 | blocked-needs-human / dependency | Direct merge still conflicts across prefix-cache/MTP/common/Qwen35 files and tests. It depends on a current-layout #237-equivalent MTP foundation before a useful port can be made. |
| #154 | blocked-needs-human / dependency | Direct merge conflicts in old `dflash/CMakeLists.txt`, MTP docs, old internal/target graph paths, and MTP smoke/contract tests. Portable only after the current-layout Qwen35 MTP foundation exists. |
| #153 | blocked-needs-human / dependency | Same old-layout MTP integrated runtime conflict class as #154; requires current-layout loader/graph/cache design work. |
| #137 | suggested-close/superseded | Direct merge conflicts only in legacy `dflash/CMakeLists.txt`; current `server/CMakeLists.txt` owns CUDA arch/BSA handling. |
| #135 | blocked-needs-human / selective-port | Direct merge conflicts in old internal/Qwen35 target graph/test paths. Needs current multi-request scheduler design rather than direct conflict resolution. |
| #94 | suggested-close/superseded | Direct merge conflicts in draft graph/safetensors loader/old internal paths. Current code already carries the useful SWA draft parsing and causal-mask behavior; remaining branch edits are old-layout. |
| #48 | suggested-close/superseded | Direct merge conflicts only in legacy `dflash/CMakeLists.txt`; current server CMake supersedes the old Blackwell arch detection patch. |

Direct-probe log: `/tmp/luce-merge-probes-20260528-162440.txt`.
Worktree used for this refresh: `/tmp/luce-auto-cron-20260528-162440`.

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Current status | Next useful action |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Define current `server/` MTP contracts first: generic vs qwen35-only backend capability, hidden-capture semantics, rollback with DDTree/spec-decode, CLI precedence among draft/remote/pflash/lazy/MTP, then port interfaces, Qwen35 capture, MTP module/build, CLI, and tests in layers. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Revisit after #237-equivalent MTP foundation; then port hidden-state/speculator hooks, backend dispatcher, daemon wiring, and WARM tests to current `server/` layout. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | blocked-needs-human / dependency | Mine linear MTP decode semantics after current-layout Qwen35 MTP exists. |
| #153 | `xabicasa/dflash-mtp-integrated` | blocked-needs-human / dependency | Mine loader/graph/cache/test ideas after current-layout Qwen35 MTP exists. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | blocked-needs-human / selective-port | Port only after designing current multi-slot `Qwen35Backend` state and scheduler API. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | suggested-close/superseded | Ask author to close or retarget to current `server/CMakeLists.txt` if anything remains. |
| #94 | `feat/dflash-qwen36-swa-draft` | suggested-close/superseded | Useful behavior appears absorbed; ask author/maintainers whether any remaining old-layout tests should be reauthored before close. |
| #48 | `fix/consumer-blackwell-auto-detect` | suggested-close/superseded | Close or retarget to current `server/CMakeLists.txt` if still needed. |

## Draft / excluded

Draft PRs remain outside the primary non-draft integration target except for
dependency awareness: #297, #291, #290, #286, #285, #275, #249, #193, and #75.
#286 is the draft PR for the current auto-integration snapshot.

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T16:23:47-04:00 at preflight; manifest refresh timestamp 2026-05-28T16:29:05-04:00.
- Primary checkout `/home/erik/Projects/luce2` `git status --short` was clean before work began.
- `git branch --show-current` reported `auto-integration` in the primary checkout.
- `git remote -v` verified `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status` succeeded for account `easel` with repo/workflow scopes.
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text` succeeded for the Claude Team account.
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version` succeeded with `codex-cli 0.130.0`; `/home/linuxbrew/.linuxbrew/bin/tmux -V` reported `tmux 3.6a`.
- `git fetch --prune origin` and `git fetch --prune easel` completed separately; targeted fetches recreated open non-draft contributor PR refs.
- Worktree `/tmp/luce-auto-cron-20260528-162440` was created from `easel/auto-integration` and verified `origin/main` is already an ancestor.
- Isolated direct probes attempted `git merge --no-commit --no-ff origin/pr/<n>` for #237, #221, #154, #153, #137, #135, #94, and #48; each conflicted and was aborted in the worktree. Log: `/tmp/luce-merge-probes-20260528-162440.txt`.
- tmux-driven Codex feasibility analysis for #237 completed without file modifications and produced `/tmp/luce-pr237-codex-20260528-162440.txt`.
- Ancestor checks passed for included open non-draft contributor PR refs #298, #295, #294, #292, #289, #276, #274, #266, #152, and #142.
- `git diff --check` passed for this manifest-only refresh before commit.

## Notes

- The primary checkout is clean but its local `auto-integration` branch remains at `89f80c58`, diverged from fetched `easel/auto-integration` `ea777e0c`; the pushed `easel/auto-integration` branch remains source of truth until supervised local branch cleanup.
- Retained worktree `/tmp/luce-auto-cron-20260528-162440`, direct-probe log `/tmp/luce-merge-probes-20260528-162440.txt`, and Codex report `/tmp/luce-pr237-codex-20260528-162440.txt`.
- Prior retained worktrees, probe logs, agent reports, and configure directories remain as listed in earlier manifest revisions; cleanup is separate maintenance.
