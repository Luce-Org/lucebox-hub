# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-27T20:38:04-04:00 through 2026-05-27T20:49:41-04:00
Current stack tip before this metadata refresh: `5a76218` (now aligned with `origin/main` `4f4d82e` and local `auto-integration` / `easel/auto-integration`; this run merged upstream main and then refreshed triage).

## Included in the current stack

| PR | Head branch | State | Notes |
|---:|---|---|---|
| #284 | `fix/draft-safetensors-rope-theta` | included | Current head `697198a` is an ancestor of the refreshed stack. |
| #276 | `fix/qwen36-claude-code-tool-calling` | included | Current head `0e3c79a` is an ancestor of the refreshed stack. |
| #274 | `feat/pflash-drafter-ee7` | included | Current head `5037b28` is an ancestor of the refreshed stack. |
| #266 | `feat/harness-typed-adapters` | included | Current head `17525ea` is an ancestor of the refreshed stack. |
| #152 | `main` | included | Current head `cf735be` is an ancestor of the refreshed stack. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | included | Current head `f2fbf62` is an ancestor of the refreshed stack. |
| #174 | `split/gemma4-14-small-vram-docs` | selectively included | The docs-only VMM guidance remains ported to current `server/README.md` after the upstream README refresh. The older inherited Gemma4 commits on this branch are not cherry-picked. |
| #94 | `feat/dflash-qwen36-swa-draft` | absorbed / selectively included | Not an ancestor after the `dflash/` → `server/` migration, but its safetensors SWA config parsing and causal-mask support remain present under current draft/common code. |
| #62 | `fix/issue-55-stable-kv-pad` | absorbed / selectively included | Not an ancestor after layout migration, but the daemon reset regression test and follow-up daemon reset fixes remain carried. |
| #48 | `fix/consumer-blackwell-auto-detect` | superseded / selectively included | The old `dflash/CMakeLists.txt` change is obsolete; current `server/CMakeLists.txt` explicitly resolves CUDA architectures and carries Blackwell/GB10 handling. |
| #285 | `feat/lucebox-docker` | partially included / draft dependency | Draft PR, not part of the non-draft target, but an earlier head is carried as the Docker/CLI/bench integration dependency. Latest draft head `32961a1` is not required by any non-draft PR this run. |

## Attempted this run

| PR | Outcome | Notes |
|---:|---|---|
| #177 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr177-20260527-2020` produced conflicts in deleted legacy `dflash/CMakeLists.txt`, moved Gemma4 header/source/test paths, `server/src/errors.cpp`, and `server/src/internal.h`. A tmux-driven Claude assessment session was launched (`luce-pr177-claude-2020`) but exited without producing a report; the raw conflict shape still confirms this older Gemma4 KV-correctness split is not mechanically mergeable into the current `server/src/gemma4` architecture. |
| #153 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr153-20260527-2020` produced conflicts in legacy `dflash/CMakeLists.txt`, current `server/src/internal.h`, `server/src/qwen35/gguf_target_loader.cpp`, `server/src/qwen35/qwen35_target_graph.cpp`, and path-mapped MTP docs/tests. Codex tmux session `luce-pr153-codex-2020` wrote `/tmp/pr153-feasibility-20260527-2020.txt`: selective port is feasible, but not as conflict-marker resolution; current-layout loader/graph/token embedding/MoE/cache semantics need deliberate design. |
| #39 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr39-20260527-2020` produced broad conflicts across deleted legacy `dflash/*` files plus current draft graph/safetensors loader and MoE smoke tests. A tmux-driven Claude assessment session was launched (`luce-pr39-claude-2020`) but exited without producing a report; the branch remains an older MoE/DDTree stack that needs a survivorship review against current qwen35moe support. |
| #183 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr183-20260527-1959` produced unmerged paths across deleted legacy `dflash/CMakeLists.txt`, moved Gemma4 source/header files, `server/src/errors.cpp`, and `server/src/internal.h`. Codex tmux session `luce-pr183-codex-2003` wrote `/tmp/pr183-feasibility-20260527-1959.txt`: safe normal resolution is not feasible; PR #183 is an older Gemma4 MTP split-chain head whose direct semantic change needs a future current-layout MTP foundation port, likely after #237 or a replacement common MTP design. |
| #182 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr182-20260527-2039` conflicted on deleted legacy `dflash/CMakeLists.txt`, moved Gemma4 target/draft/MTP source and tests, `server/src/errors.cpp`, and `server/src/internal.h`. Claude exited without a report; Codex session `luce-pr182-codex2-20260527-2039` wrote `/tmp/pr182-feasibility-20260527-2039.txt`: the MTP assistant loader/step graph are portable only as a current-layout `server/src/gemma4` rewrite, not as direct conflict resolution. |
| #181 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr181-20260527-2039` hit the same old Gemma4 split-chain/current-layout boundary as #180/#182, including moved Gemma4 target/draft files, `server/src/errors.cpp`, and `server/src/internal.h`. It overlaps current Gemma4 DFlash backend and should be mined only after deciding the current Gemma4 MTP port shape. |
| #180 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr180-20260527-2039` conflicted on legacy `dflash/CMakeLists.txt`, moved Gemma4 target/draft files and tests, `server/src/errors.cpp`, and `server/src/internal.h`. Its draft-loader/quant pieces overlap current `server/src/gemma4` loader/backend work and are not mechanically mergeable. |
| #154 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr154-20260527-2039` conflicted in current Qwen35 loader/graph/test files plus old-layout MTP docs/scripts/tests. Codex session `luce-pr154-codex-20260527-2039` wrote `/tmp/pr154-feasibility-20260527-2039.txt`: the linear native MTP decode semantics are portable with moderate risk, but require a selective current-layout Qwen35 MTP design preserving current MoE/NVFP4/server abstractions. |
| #131 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr131-20260527-2039` produced broad conflicts across `.gitmodules`, deleted `dflash/*`, current Gemma4 source/test files, `server/src/internal.h`, `server/src/errors.cpp`, and pFlash adapter/test code. Claude exited without a report; Codex session `luce-pr131-codex2-20260527-2039` wrote `/tmp/pr131-feasibility-20260527-2039.txt`: broad Gemma4 support can be mined, but direct merge would reintroduce the old `dflash27b` API and obsolete build layout. |
| upstream sync | updated | `origin/main` advanced to `4f4d82e`; `easel/auto-integration` was refreshed from that tip during this run. |

## Previously attempted / still not integrated

| PR | Head branch | State | Why held |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Prior fresh worktree and Codex attempts found broad conflicts in server CMake/backend/MTP/Qwen35 graph/test code. Valuable, but needs a deliberate current-layout MTP foundation port rather than conflict-marker resolution. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Prior direct + Codex attempts left many unmerged paths and legacy artifacts; depends on a coherent #237-style MTP foundation first. |
| #183 | `split/gemma4-11a-target-mtp-integration` | blocked-needs-human / superseded-by-current-architecture | Revalidated in the previous run. Its direct `h_prev` assertion has no current target location until Gemma4 MTP runtime is reintroduced in the current `server/src/gemma4` architecture. |
| #182 | `split/gemma4-10-mtp-loader-step-graph` | attempted / selective-port | Revalidated this run in `/tmp/luce-attempt-pr182-20260527-2039`; Codex report says the assistant loader/MTP graph are portable only as current-layout `server/src/gemma4` work. |
| #181 | `split/gemma4-09-dflash-draft-runtime` | attempted / likely superseded | Revalidated this run in `/tmp/luce-attempt-pr181-20260527-2039`; conflicts overlap current Gemma4 DFlash backend and the #182 MTP port decision. |
| #180 | `split/gemma4-08-draft-loader-quant` | attempted / likely superseded | Revalidated this run in `/tmp/luce-attempt-pr180-20260527-2039`; conflicts overlap current Gemma4 loader/backend work and are not mechanically mergeable. |
| #177 | `split/gemma4-06-kv-correctness` | attempted / not mechanically mergeable | Revalidated this run in `/tmp/luce-attempt-pr177-20260527-2020`; conflicts are on the old Gemma4 split-chain/current-layout boundary. Needs selective correctness review, not a branch merge. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | attempted / selective-port | Revalidated this run in `/tmp/luce-attempt-pr154-20260527-2039`; Codex report says the linear native MTP decode semantics are portable with moderate risk, but require current-layout Qwen35 MTP design work. |
| #153 | `xabicasa/dflash-mtp-integrated` | blocked-needs-human / selective-port | Revalidated this run in `/tmp/luce-attempt-pr153-20260527-2020`; Codex report says a current-layout Qwen35 MTP port is feasible but needs loader/graph/cache/MoE design rather than a mechanical merge. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | blocked-needs-human / stale | Legacy `dflash/CMakeLists.txt` only; current `server/CMakeLists.txt` supersedes it. Suggested close unless re-targeted to current server layout. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | blocked-needs-human / selective-port | Prior manual and Codex review concluded the concept requires a current-tree scheduler redesign around `HttpServer`/`ModelBackend`/Qwen35 cache APIs. |
| #131 | `feature/gemma4-support` | attempted / selective-port | Revalidated this run in `/tmp/luce-attempt-pr131-20260527-2039`; Codex report says broad Gemma4 support can be mined, but direct merge would resurrect old `dflash27b`/`dflash/` layout. |
| #39 | `feat/moe-35b-a3b` | attempted / likely superseded | Revalidated this run in `/tmp/luce-attempt-pr39-20260527-2020`; direct merge conflicts with current qwen35moe/draft graph work. Needs a MoE/DDTree survivorship review rather than a branch merge. |

## Draft / excluded

Draft PRs remain outside the primary contributor integration target: #289, #286, #285, #278, #275, #273, #265, #249, #193, #75. #285 remains partially carried only as an integration dependency.

## Suggested close / author action

- #137: close or ask author to re-target current `server/CMakeLists.txt`.
- #48: close as superseded by current CUDA architecture/Blackwell handling unless the author has a specific missing Blackwell case.
- #183/#182/#181/#180/#177/#154/#153/#131/#39: do not close solely as conflicts yet; ask authors or maintainers to identify which current-layout Gemma4/MTP pieces are still intended after the newer `server/src/gemma4` architecture and #237 direction.

## Validation run

This run performed:

- `git merge --no-edit origin/main` to refresh the branch against the latest upstream main (`4f4d82e`)
- `date -Is`
- `git status --short`, `git remote -v`, branch and revision checks in the primary checkout
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status`
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text`
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version`
- `git fetch --prune origin`
- `git fetch --prune easel`
- `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 200 --json ... --jq ...`
- Targeted fetches for all open non-draft PR refs (`origin/pr/<n>`)
- `git cherry easel/auto-integration origin/pr/<n>` and `git merge-base --is-ancestor origin/pr/<n> easel/auto-integration` classification checks
- Isolated metadata worktree `/tmp/luce-auto-cron-20260527-2020`
- Fresh isolated PR #177 merge probe in `/tmp/luce-attempt-pr177-20260527-2020`
- Fresh isolated PR #153 merge probe in `/tmp/luce-attempt-pr153-20260527-2020`
- Fresh isolated PR #39 merge probe in `/tmp/luce-attempt-pr39-20260527-2020`
- tmux-driven Claude assessment sessions `luce-pr177-claude-2020` and `luce-pr39-claude-2020` (both exited before producing report files)
- tmux-driven Codex assessment session `luce-pr153-codex-2020`, producing `/tmp/pr153-feasibility-20260527-2020.txt`
- `git diff --check -- docs/auto-integration.md` before metadata commit (clean)
- Fresh isolated PR #182 merge probe in `/tmp/luce-attempt-pr182-20260527-2039`
- Fresh isolated PR #181 merge probe in `/tmp/luce-attempt-pr181-20260527-2039`
- Fresh isolated PR #180 merge probe in `/tmp/luce-attempt-pr180-20260527-2039`
- Fresh isolated PR #154 merge probe in `/tmp/luce-attempt-pr154-20260527-2039`
- Fresh isolated PR #131 merge probe in `/tmp/luce-attempt-pr131-20260527-2039`
- tmux-driven Claude assessment sessions `luce-pr182-claude-20260527-2039` and `luce-pr131-claude-20260527-2039` (both exited before producing report files)
- tmux-driven Codex assessment sessions `luce-pr182-codex2-20260527-2039`, `luce-pr154-codex-20260527-2039`, and `luce-pr131-codex2-20260527-2039`, producing `/tmp/pr182-feasibility-20260527-2039.txt`, `/tmp/pr154-feasibility-20260527-2039.txt`, and `/tmp/pr131-feasibility-20260527-2039.txt`
- `git diff --check HEAD^1..HEAD` after the upstream merge and metadata refresh (clean)
- `uv run --directory luce-bench --with pytest pytest -q tests/test_runner.py tests/test_smoke_area.py` (passed in this run)

## Notes

- Primary checkout `/home/erik/Projects/luce2` was clean at start and was not edited directly during reconciliation.
- Retained worktree `/tmp/luce-auto-cron-20260527-2020` for this metadata refresh until pushed/verified.
- Retained conflicted worktree `/tmp/luce-attempt-pr177-20260527-2020` for audit; it contains unmerged PR #177 probe state.
- Retained conflicted worktree `/tmp/luce-attempt-pr153-20260527-2020` for audit; it contains unmerged PR #153 probe state and corresponds to `/tmp/pr153-feasibility-20260527-2020.txt`.
- Retained conflicted worktree `/tmp/luce-attempt-pr39-20260527-2020` for audit; it contains unmerged PR #39 probe state.
- Retained conflicted worktree `/tmp/luce-attempt-pr182-20260527-2039` for audit; it contains unmerged PR #182 probe state and corresponds to `/tmp/pr182-feasibility-20260527-2039.txt`.
- Retained conflicted worktree `/tmp/luce-attempt-pr181-20260527-2039` for audit; it contains unmerged PR #181 probe state.
- Retained conflicted worktree `/tmp/luce-attempt-pr180-20260527-2039` for audit; it contains unmerged PR #180 probe state.
- Retained conflicted worktree `/tmp/luce-attempt-pr154-20260527-2039` for audit; it contains unmerged PR #154 probe state and corresponds to `/tmp/pr154-feasibility-20260527-2039.txt`.
- Retained conflicted worktree `/tmp/luce-attempt-pr131-20260527-2039` for audit; it contains unmerged PR #131 probe state and corresponds to `/tmp/pr131-feasibility-20260527-2039.txt`.
- Retained prior conflicted worktrees from earlier runs as previously reported; cleanup would be separate maintenance.
