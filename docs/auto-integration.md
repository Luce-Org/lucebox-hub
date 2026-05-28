# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T02:45:38-04:00
Current base: `origin/main` `4f4d82e`
Current integration tip before this refresh: `easel/auto-integration` `cf468ab`

This branch is maintained as a reproducible patch stack over `origin/main`. The
primary checkout was clean at the start of this unattended run. Upstream
`origin/main` and all currently included non-draft contributor PR heads were
unchanged relative to the prior pushed stack, so this refresh did not add source
code. The remaining non-ancestor non-draft PRs were re-probed in the isolated
integration worktree; every direct merge still conflicts. PR #237 received a
fresh tmux-driven Codex feasibility pass in a conflicted worktree, which
confirmed it is feasible only as a deliberate current-layout MTP port rather
than a conflict-marker resolution.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #284 | `fix/draft-safetensors-rope-theta` | `63bba30` | included | Current head is an ancestor of the refreshed stack; rejects non-finite `rope_theta` values from draft safetensors `config.json`. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `0e3c79a` | included | Current head is an ancestor of the refreshed stack. |
| #274 | `feat/pflash-drafter-ee7` | `5037b28` | included | Current head is an ancestor of the refreshed stack. |
| #266 | `feat/harness-typed-adapters` | `17525ea` | included | Current head is an ancestor of the refreshed stack. |
| #152 | `main` | `cf735be` | included | Current head is an ancestor of the refreshed stack. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | `f2fbf62` | included | Current head is an ancestor of the refreshed stack. |
| #174 | `split/gemma4-14-small-vram-docs` | `8b1caba` | selectively included | The useful small-VRAM/VMM documentation is already ported into the current `server/README.md`; the remaining old Gemma4 split-chain commits are not ancestors. |
| #94 | `feat/dflash-qwen36-swa-draft` | `d2f9c9d` | absorbed / selectively included | Current draft/common code already carries the safetensors SWA config parsing and causal-mask behavior; the old branch conflicts on moved/rewritten draft files. |
| #62 | `fix/issue-55-stable-kv-pad` | `0ce6832` | absorbed / selectively included | Daemon reset regression coverage and reset fixes remain carried in the current server layout; the old branch still conflicts on legacy test paths. |
| #48 | `fix/consumer-blackwell-auto-detect` | `858b84b` | superseded / selectively included | Current `server/CMakeLists.txt` owns CUDA architecture resolution and Blackwell/GB10 handling; the PR edits deleted legacy `dflash/CMakeLists.txt`. |
| #39 | `feat/moe-35b-a3b` | `c86ec86` | partially integrated / mostly superseded | Prior run ported draft safetensors `config.json`/YaRN survivorship into current `server/src/draft/*`; the remaining loader/FFN/target-graph pieces are superseded by current qwen35moe/DDTree code. |
| #285 | `feat/lucebox-docker` | draft | partial draft dependency | Draft PR, outside the non-draft target, but an earlier Docker/CLI/bench integration dependency remains carried. |

Recently closed contributor PRs #287 and #288 are no longer open non-draft
integration targets; their heads remain in the historical stack where already
carried.

## Attempted this run

| PR | Outcome | Notes |
|---:|---|---|
| upstream sync | checked | In isolated worktree `/tmp/luce-auto-cron-20260528-023928`, `git merge --no-edit origin/main` reported `Already up to date.` |
| all remaining non-ancestor non-draft PRs | direct merge probes still conflicted | Fresh isolated direct probes attempted `--no-commit --no-ff` merges for #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39. Every direct probe conflicted and was aborted in the isolated worktree. Consolidated output: `/tmp/luce-merge-probes-20260528-023928.txt`. |
| #237 | fresh delegated/manual feasibility retained | Conflicted probe worktree `/tmp/luce-pr237-feas-20260528-023928` remains for inspection. Codex completed in tmux and produced `/tmp/pr237-codex-feasibility-20260528-023928.txt`, confirming the portable value is the native Qwen35/Qwen3.6 MTP stack (`mtp_interface`, chain runner/orchestrator, MTP GGUF metadata, hidden capture, backend path, native server flags, and tests). It should be ported into `server/` in layers rather than by accepting the conflicted merge. |

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Current status | Next useful action |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Direct merge probe still conflicts. Fresh Codex report `/tmp/pr237-codex-feasibility-20260528-023928.txt` recommends: port additive MTP files/CMake first; merge `ModelBackend`, `DFlashTarget`, `StepGraph`, and `BackendArgs` additively; preserve current remote-draft, PFlash, qwen35moe, and reasoning-budget paths; then add optional Qwen35 backend/native-server MTP flags and tests. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Should wait for #237/equivalent MTP foundation; then selectively port unique dual-speculator routing, MTP head-KV WARM snapshot/restore, partial `warm_head_kv_range`, native HTTP/daemon `speculator` request plumbing, and focused prefix-cache MTP tests. |
| #183 | `split/gemma4-11a-target-mtp-integration` | blocked-needs-human / current-layout design required | Unique value is Gemma4 assistant/MTP graph and target hidden-state capture/asymmetric KV tests, but it is old-layout and overlaps current `server/src/gemma4/*`; port only after deciding how Gemma4 MTP composes with the existing feature-complete Gemma4 backend/DFlash target. |
| #182 | `split/gemma4-10-mtp-loader-step-graph` | selective-port | Portable only as current-layout `server/src/gemma4` assistant loader/MTP graph work. Prior Codex report: `/tmp/pr182-feasibility-20260527-2039.txt`. |
| #181 | `split/gemma4-09-dflash-draft-runtime` | likely superseded / selective-port | Overlaps current Gemma4 DFlash backend and should be mined only after the current Gemma4 MTP shape is chosen. |
| #180 | `split/gemma4-08-draft-loader-quant` | likely superseded / selective-port | Overlaps current Gemma4 loader/backend code; not mechanically mergeable. |
| #177 | `split/gemma4-06-kv-correctness` | selective-port / blocked-needs-human | Old-layout additions `dflash/include/gemma4.h`, `dflash/src/gemma4_target_graph.cpp`, `dflash/src/gemma4_target_loader.cpp`, and Gemma4 tests need deliberate mapping into current `server/` APIs. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | selective-port | Linear native MTP decode semantics are portable with moderate risk after a current-layout Qwen35 MTP design. Prior Codex report: `/tmp/pr154-feasibility-20260527-2039.txt`. |
| #153 | `xabicasa/dflash-mtp-integrated` | blocked-needs-human / selective-port | Current-layout Qwen35 MTP port is feasible but requires loader/graph/cache/MoE design, not conflict-marker resolution. Prior Codex report: `/tmp/pr153-feasibility-20260527-2020.txt`. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | stale / suggested close | Edits only deleted legacy `dflash/CMakeLists.txt`; close or ask author to retarget current `server/CMakeLists.txt`. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | blocked-needs-human / selective-port | Prior Codex report: `/tmp/pr135-codex-feasibility-20260528-020651.txt`. Port order: add `n_seqs` primitives while preserving current HEAD behavior, port Qwen35 batched graph/probe path as comparison-only, move aligned-bucket helper/selftest into current server tests, then design multi-slot `Qwen35Backend` state and scheduler API before production exposure. |
| #131 | `feature/gemma4-support` | selective-port | Broad Gemma4 support can be mined, but direct merge would resurrect old `dflash27b`/`dflash/` layout. Prior Codex report: `/tmp/pr131-feasibility-20260527-2039.txt`. |
| #94 | `feat/dflash-qwen36-swa-draft` | absorbed / verify-only | Remaining conflicts are old-layout draft/common paths; current code carries the useful SWA pieces. |
| #62 | `fix/issue-55-stable-kv-pad` | absorbed / verify-only | Remaining conflicts are old-layout tests; current code carries reset behavior. |
| #48 | `fix/consumer-blackwell-auto-detect` | superseded / suggested close | Current CMake supersedes the old `dflash/CMakeLists.txt` change. |
| #39 | `feat/moe-35b-a3b` | partially integrated / mostly superseded | Ask maintainers whether any remaining old MoE smoke coverage should be reauthored against current qwen35moe APIs before closing. Prior Codex report: `/tmp/pr39-survivorship-20260527-222159.txt`. |

## Draft / excluded

Draft PRs remain outside the primary non-draft integration target except for
dependency awareness: #289, #286, #285, #278, #275, #273, #265, #249, #193,
and #75. #289 (`feat(qwen35moe): pipelined hybrid MoE decode with GPU/CPU
overlap`) remains draft-only and was not attempted. #285 remains partially
carried only as an integration dependency.

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T02:38:38-04:00 during preflight, 2026-05-28T02:45:38-04:00 while refreshing this manifest.
- Primary checkout `git status --short` was clean before work began and stayed clean while probing in worktrees.
- `git remote -v` verified `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status` succeeded for account `easel`.
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text` succeeded for the Claude Team account.
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version` succeeded (`codex-cli 0.130.0`).
- `git fetch --prune origin` and `git fetch --prune easel` completed; targeted fetches recreated current open non-draft PR refs.
- `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 200 --json ... --jq ...` enumerated all open PRs.
- `git merge-base --is-ancestor origin/pr/<n> <stack>` classification checks: #284, #276, #274, #266, #152, and #142 are current ancestors; #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39 remain non-ancestor/selective-port candidates.
- Isolated reconciliation/probe worktree `/tmp/luce-auto-cron-20260528-023928`; `git merge --no-edit origin/main` reported already up to date.
- Fresh direct merge probes for all remaining non-ancestor non-draft PR refs: #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39; all direct probes conflicted and were aborted in the isolated worktree; consolidated output retained at `/tmp/luce-merge-probes-20260528-023928.txt`.
- tmux-driven Codex feasibility pass for #237 completed in `/tmp/luce-pr237-feas-20260528-023928`; report retained at `/tmp/pr237-codex-feasibility-20260528-023928.txt`.
- `git diff --check origin/main...HEAD` reported only pre-existing whitespace warnings in `luce-bench/src/lucebench/fixtures/forge_eval/scenarios/_model_quality.py` and `_stateful_model_quality.py`.
- `git diff --check HEAD~1..HEAD` passed for this run's manifest-only change.
- `cmake -S server -B /tmp/luce-build-20260528-023928 -DLUCE_BUILD_TESTS=ON` still fails before project compilation in this WSL CUDA setup because compiler identification selects unsupported `sm_52` (`ptxas fatal: Value 'sm_52' is not defined for option 'gpu-name'`).

## Notes

- Primary checkout `/home/erik/Projects/luce2` stayed clean during probing and was modified only after the verified manifest refresh was ready.
- Retained worktree `/tmp/luce-auto-cron-20260528-023928` for direct-merge probe audit.
- Retained conflicted PR #237 feasibility worktree `/tmp/luce-pr237-feas-20260528-023928` and Codex report `/tmp/pr237-codex-feasibility-20260528-023928.txt`.
- Retained direct-probe log `/tmp/luce-merge-probes-20260528-023928.txt`.
- Retained CMake configure directory `/tmp/luce-build-20260528-023928` for inspection of the local CUDA compiler-identification failure.
- Prior retained conflicted worktrees and agent reports remain as listed in earlier manifest revisions; cleanup is separate maintenance.
