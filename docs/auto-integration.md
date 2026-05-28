# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T00:53:49-04:00
Current base: `origin/main` `4f4d82e`
Current integration tip before this refresh: `easel/auto-integration` `6599b09`

This branch is maintained as a reproducible patch stack over `origin/main`. The
primary checkout was clean at the start of this unattended run. This refresh did
not find an upstream-base advance or a newly mergeable non-draft PR head, but it
reran direct merge probes for all remaining non-ancestor non-draft PRs and added
fresh conflicted-worktree/delegated evidence for PR #183.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #284 | `fix/draft-safetensors-rope-theta` | `697198a` | included | Current head is an ancestor of `easel/auto-integration`. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `0e3c79a` | included | Current head is an ancestor of `easel/auto-integration`. |
| #274 | `feat/pflash-drafter-ee7` | `5037b28` | included | Current head is an ancestor of `easel/auto-integration`. |
| #266 | `feat/harness-typed-adapters` | `17525ea` | included | Current head is an ancestor of `easel/auto-integration`. |
| #152 | `main` | `cf735be` | included | Current head is an ancestor of `easel/auto-integration`. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | `f2fbf62` | included | Current head is an ancestor of `easel/auto-integration`. |
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
| upstream sync | checked | In isolated worktree `/tmp/luce-auto-cron-20260528-004553`, `git merge --no-edit origin/main` reported `Already up to date.` |
| all non-ancestor non-draft PRs | direct merge probes still conflicted | Fresh isolated direct probes attempted `--no-commit --no-ff` merges for #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39. Every direct probe conflicted and was aborted in the isolated worktree. Consolidated output: `/tmp/luce-merge-probes-20260528-004553.txt`. |
| #237 | delegated/manual feasibility attempt | Previous run created conflicted probe worktree `/tmp/luce-pr237-feas-20260528-000006`. A tmux-driven Claude feasibility run exited with only `Error: Reached max turns (12)` (`/tmp/pr237-claude-feasibility-20260528-000006.txt`), so no Claude conclusion is claimed. A tmux-driven Codex feasibility run completed (`/tmp/pr237-codex-feasibility-20260528-000006.txt`) and confirmed #237 is not mechanically mergeable: 24 unmerged paths, stale deleted `dflash/` server paths, and semantic conflicts in MTP source selection, qwen35/qwen35moe GGUF metadata, graph-builder hidden-state capture vs MoE router capture, remote-draft/PFlash coexistence, and CMake/test wiring. Codex recommends a deliberate current-layout selective port of `server/src/common/mtp_*`, `server/src/common/gguf_metadata.h`, `server/src/qwen35/qwen35_mtp*`, and `server/test/test_common_mtp_orchestrator.cpp`, plus targeted merges into backend factory, graph builders, loader, daemon/backend, native server CLI, and tests. |
| #221 | delegated/manual feasibility attempt | Created conflicted probe worktree `/tmp/luce-pr221-feas-20260528-002537`. A tmux-driven Claude feasibility run exited with only `Error: Reached max turns (18)` (`/tmp/pr221-claude-feasibility-20260528-002537.txt`), so no Claude conclusion is claimed. A tmux-driven Codex feasibility run had to be stopped after producing a large report (`/tmp/pr221-codex-feasibility-20260528-002537.txt`); the usable final section confirms #221 is valuable but not mechanically resolvable: 36 unmerged paths, old `dflash/` control-plane files, PR-only MTP/benchmark additions, and semantic conflicts around dual-speculator dispatch, MTP head-KV WARM snapshot/restore, current native HTTP/daemon request fields, MoE metadata, graph hidden capture, and prefix-cache integration. Codex recommends landing/porting #237's MTP foundation first, then selectively porting #221's dual-dispatch and prefix-WARM overlay into current `server/` files. |
| #183 | delegated/manual feasibility attempt | Created conflicted probe worktree `/tmp/luce-pr183-feas-20260528-004611`. Direct merge shows deleted legacy `dflash/CMakeLists.txt`, file-location conflicts for old-layout `dflash/include/gemma4.h` and `dflash/src/gemma4_*`, content conflicts in `server/src/errors.cpp` and `server/src/internal.h`, and old-layout Gemma4 smoke/MTP tests. Tmux-driven Claude exited with only `Error: Reached max turns (18)` (`/tmp/pr183-claude-feasibility-20260528-004611.txt`), so no Claude conclusion is claimed. Tmux-driven Codex was stopped after producing large read-only inspection output without a final recommendation (`/tmp/pr183-codex-feasibility-20260528-004611.txt`). Manual inspection confirms #183's unique value is Gemma4 target-graph MTP integration (assistant loader, MTP graph, target graph hidden-state capture/asymmetric KV, and tests), but current `server/src/gemma4/*` already has a different feature-complete Gemma4 backend/DFlash target shape; safe integration requires a deliberate current-layout Gemma4 MTP design rather than conflict-marker resolution. |
| #177 | delegated/manual feasibility attempt | Previous run created conflicted probe worktree `/tmp/luce-pr177-feas-20260527-234303` and attempted two tmux-driven Claude read-only feasibility runs. Both exited with only `Error: Reached max turns` (`/tmp/pr177-claude-feasibility-20260527-234314.txt`, `/tmp/pr177-claude-feasibility-20260527-234433.txt`), so no delegated conclusion is claimed. Manual diff/status inspection shows #177 is a 9-file / ~2.5k-line old-layout Gemma4 KV/target-graph slice that maps mostly from deleted `dflash/*` into current `server/*` and conflicts in `server/src/errors.cpp` and `server/src/internal.h`; it remains a selective current-layout Gemma4 correctness port, not a mechanical merge. |

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Current status | Next useful action |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Direct merge probe still conflicts; previous delegated worktree/Codex report remains the current guidance. Do not resolve by accepting one side. Minimal viable current-layout port starts with `server/src/common/mtp_*`, `server/src/common/gguf_metadata.h`, `server/src/qwen35/qwen35_mtp*`, and `server/test/test_common_mtp_orchestrator.cpp`, then deliberate merges into backend factory, qwen35 graph builders/loader/backend/daemon, native server CLI, and CMake/CTest while preserving qwen35moe, remote-draft, PFlash, and token-callback behavior. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Current run revalidated with a conflicted worktree and Codex feasibility report. #221 should wait for #237/equivalent MTP foundation; then selectively port its unique dual-speculator routing, MTP head-KV WARM snapshot/restore, partial `warm_head_kv_range`, native HTTP/daemon `speculator` request plumbing, and focused prefix-cache MTP tests. Do not restore deleted `dflash/scripts/*` or `dflash/src/qwen36/*` paths. |
| #183 | `split/gemma4-11a-target-mtp-integration` | blocked-needs-human / current-layout design required | Current run revalidated with conflicted worktree `/tmp/luce-pr183-feas-20260528-004611`. Unique value is Gemma4 assistant/MTP graph and target hidden-state capture/asymmetric KV tests, but it is old-layout and overlaps current `server/src/gemma4/*`; port only after deciding how Gemma4 MTP composes with the existing feature-complete Gemma4 backend/DFlash target. |
| #182 | `split/gemma4-10-mtp-loader-step-graph` | selective-port | Portable only as current-layout `server/src/gemma4` assistant loader/MTP graph work. Prior Codex report: `/tmp/pr182-feasibility-20260527-2039.txt`. |
| #181 | `split/gemma4-09-dflash-draft-runtime` | likely superseded / selective-port | Overlaps current Gemma4 DFlash backend and should be mined only after the current Gemma4 MTP shape is chosen. |
| #180 | `split/gemma4-08-draft-loader-quant` | likely superseded / selective-port | Overlaps current Gemma4 loader/backend code; not mechanically mergeable. |
| #177 | `split/gemma4-06-kv-correctness` | selective-port / blocked-needs-human | Current run revalidated conflict shape: old-layout additions `dflash/include/gemma4.h`, `dflash/src/gemma4_target_graph.cpp`, `dflash/src/gemma4_target_loader.cpp`, and Gemma4 tests need deliberate mapping into current `server/` APIs; no safe standalone conflict resolution was found. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | selective-port | Linear native MTP decode semantics are portable with moderate risk after a current-layout Qwen35 MTP design. Prior Codex report: `/tmp/pr154-feasibility-20260527-2039.txt`. |
| #153 | `xabicasa/dflash-mtp-integrated` | blocked-needs-human / selective-port | Current-layout Qwen35 MTP port is feasible but requires loader/graph/cache/MoE design, not conflict-marker resolution. Prior Codex report: `/tmp/pr153-feasibility-20260527-2020.txt`. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | stale / suggested close | Edits only deleted legacy `dflash/CMakeLists.txt`; close or ask author to retarget current `server/CMakeLists.txt`. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | selective-port | Low-level aligned-bucket and `n_seqs` mechanics need deliberate current `ModelBackend`/Qwen35Backend graph/API work. Prior Codex report: `/tmp/pr135-feasibility-20260527-2203.txt`. |
| #131 | `feature/gemma4-support` | selective-port | Broad Gemma4 support can be mined, but direct merge would resurrect old `dflash27b`/`dflash/` layout. Prior Codex report: `/tmp/pr131-feasibility-20260527-2039.txt`. |
| #94 | `feat/dflash-qwen36-swa-draft` | absorbed / verify-only | Remaining conflicts are old-layout draft/common paths; current code carries the useful SWA pieces. |
| #62 | `fix/issue-55-stable-kv-pad` | absorbed / verify-only | Remaining conflicts are old-layout tests; current code carries reset behavior. |
| #48 | `fix/consumer-blackwell-auto-detect` | superseded / suggested close | Current CMake supersedes the old `dflash/CMakeLists.txt` change. |
| #39 | `feat/moe-35b-a3b` | partially integrated / mostly superseded | Ask maintainers whether any remaining old MoE smoke coverage should be reauthored against current qwen35moe APIs before closing. Prior Codex report: `/tmp/pr39-survivorship-20260527-222159.txt`. |

## Draft / excluded

Draft PRs remain outside the primary non-draft integration target except for
dependency awareness: #289, #286, #285, #278, #275, #273, #265, #249, #193,
and #75. #285 remains partially carried only as an integration dependency.

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T00:45:12-04:00 during preflight, 2026-05-28T00:53:49-04:00 for this metadata refresh.
- Primary checkout `git status --short` was clean before work began and remained clean while probing in worktrees.
- `git remote -v` verified `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status` succeeded for account `easel`.
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text` succeeded for the Claude Team account.
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --help` succeeded; `/home/linuxbrew/.linuxbrew/bin/tmux -V` reported `tmux 3.6a`.
- `git fetch --prune origin` and `git fetch --prune easel` completed; targeted fetches recreated current open non-draft PR refs.
- `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 200 --json ... --jq ...` enumerated all open PRs.
- Targeted fetches for all open non-draft PR refs (`origin/pr/<n>`).
- `git merge-base --is-ancestor origin/pr/<n> easel/auto-integration` classification checks: #284, #276, #274, #266, #152, and #142 are current ancestors; #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39 remain non-ancestor/selective-port candidates.
- Isolated reconciliation/probe worktree `/tmp/luce-auto-cron-20260528-004553`; `git merge --no-edit origin/main` reported already up to date.
- Fresh direct merge probes for all currently non-ancestor non-draft PR refs: #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39; all direct probes conflicted and were aborted in the isolated worktree; consolidated output retained at `/tmp/luce-merge-probes-20260528-004553.txt`.
- Tmux-driven Claude delegation for #183 was monitorable but exited with only `Error: Reached max turns (18)`; this is recorded above rather than treated as a successful delegated conclusion.
- Tmux-driven Codex delegation for #183 had to be interrupted after producing large read-only inspection output without a final recommendation; report retained at `/tmp/pr183-codex-feasibility-20260528-004611.txt`.
- Manual read-only `git status --short`, conflict output, and current `server/src/gemma4/*` inspection confirmed #183 remains a current-layout Gemma4 MTP design/port rather than a mechanical merge.

No source/build validation was rerun because this refresh changes only
integration metadata. Previous source validation remains unchanged: local
CMake/CUDA configure fails before project compilation in this WSL CUDA setup
because compiler identification selects unsupported `sm_52` (`ptxas fatal:
Value 'sm_52' is not defined for option 'gpu-name'`).

## Notes

- Primary checkout `/home/erik/Projects/luce2` stayed clean during probing and was modified only after the verified metadata commit was ready.
- Retained worktree `/tmp/luce-auto-cron-20260528-004553` for this metadata refresh and direct-merge probe audit.
- Retained conflicted probe worktree `/tmp/luce-pr183-feas-20260528-004611` because it contains conflict state useful for a future Gemma4 MTP selective-port attempt.
- Retained delegated report/error paths `/tmp/pr183-claude-feasibility-20260528-004611.txt` and `/tmp/pr183-codex-feasibility-20260528-004611.txt`.
- Prior retained conflicted worktrees and agent reports remain as listed in earlier manifest revisions; cleanup is separate maintenance.
