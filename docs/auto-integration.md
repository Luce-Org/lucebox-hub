# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-27T19:59:25-04:00 through 2026-05-27T20:08:00-04:00
Current stack tip before this metadata refresh: `75eea4a` (merged latest `origin/main` `3474edf` into prior `easel/auto-integration` `4eae787`).

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
| #183 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr183-20260527-1959` produced unmerged paths across deleted legacy `dflash/CMakeLists.txt`, moved Gemma4 source/header files, `server/src/errors.cpp`, and `server/src/internal.h`. Codex tmux session `luce-pr183-codex-2003` wrote `/tmp/pr183-feasibility-20260527-1959.txt`: safe normal resolution is not feasible; PR #183 is an older Gemma4 MTP split-chain head whose direct semantic change needs a future current-layout MTP foundation port, likely after #237 or a replacement common MTP design. |
| upstream sync | integrated | Merged latest `origin/main` (`3474edf`) into the integration worktree. Conflicts were limited to README docs; resolution kept upstream's condensed root README and retained the previously ported small-VRAM VMM guidance in `server/README.md`. |

## Previously attempted / still not integrated

| PR | Head branch | State | Why held |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Prior fresh worktree and Codex attempts found broad conflicts in server CMake/backend/MTP/Qwen35 graph/test code. Valuable, but needs a deliberate current-layout MTP foundation port rather than conflict-marker resolution. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Prior direct + Codex attempts left many unmerged paths and legacy artifacts; depends on a coherent #237-style MTP foundation first. |
| #183 | `split/gemma4-11a-target-mtp-integration` | blocked-needs-human / superseded-by-current-architecture | Revalidated this run. Its direct `h_prev` assertion has no current target location until Gemma4 MTP runtime is reintroduced in the current `server/src/gemma4` architecture. |
| #182 | `split/gemma4-10-mtp-loader-step-graph` | pending / likely superseded | Older Gemma4 split chain; should be assessed only after the MTP foundation decision. |
| #181 | `split/gemma4-09-dflash-draft-runtime` | pending / likely superseded | Older Gemma4 split chain; overlaps current Gemma4 DFlash backend. |
| #180 | `split/gemma4-08-draft-loader-quant` | pending / likely superseded | Older Gemma4 split chain; overlaps current loader/backend work. |
| #177 | `split/gemma4-06-kv-correctness` | pending / selective-correctness-review | Older Gemma4 split chain; may contain correctness details worth porting, but not a safe branch merge. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | pending / likely superseded | Older dFlash MTP line stacked on #153; likely superseded by #237 or replacement MTP foundation. |
| #153 | `xabicasa/dflash-mtp-integrated` | pending / likely superseded | Older dFlash MTP line; likely superseded by #237. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | blocked-needs-human / stale | Legacy `dflash/CMakeLists.txt` only; current `server/CMakeLists.txt` supersedes it. Suggested close unless re-targeted to current server layout. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | blocked-needs-human / selective-port | Prior manual and Codex review concluded the concept requires a current-tree scheduler redesign around `HttpServer`/`ModelBackend`/Qwen35 cache APIs. |
| #131 | `feature/gemma4-support` | pending / likely superseded | Broad older Gemma4 support stack; overlaps later current Gemma4 work and draft #193. |
| #39 | `feat/moe-35b-a3b` | pending / likely superseded | Older MoE/draft work conflicts with newer stacks; needs a separate survivorship review. |

## Draft / excluded

Draft PRs remain outside the primary contributor integration target: #286, #285, #278, #275, #273, #265, #249, #193, #75. #285 remains partially carried only as an integration dependency.

## Suggested close / author action

- #137: close or ask author to re-target current `server/CMakeLists.txt`.
- #48: close as superseded by current CUDA architecture/Blackwell handling unless the author has a specific missing Blackwell case.
- #183/#182/#181/#180/#177/#154/#153/#131/#39: do not close solely as conflicts yet; ask authors or maintainers to identify which current-layout Gemma4/MTP pieces are still intended after the newer `server/src/gemma4` architecture and #237 direction.

## Validation run

This run performed:

- `date -Is`
- `git status --short`, `git remote -v`, branch and revision checks in the primary checkout
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status`
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text`
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version`
- `git fetch --prune origin`
- `git fetch --prune easel`
- `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 200 --json ...`
- Targeted fetches for all open non-draft PR refs (`origin/pr/<n>`)
- `git cherry HEAD origin/pr/<n>` and `git merge-base --is-ancestor origin/pr/<n> HEAD` classification checks
- Isolated reconciliation worktree `/tmp/luce-auto-cron-20260527-1959`
- `git merge --no-edit origin/main`, with README conflict resolution
- Fresh isolated PR #183 merge probe in `/tmp/luce-attempt-pr183-20260527-1959`
- tmux-driven Codex assessment session `luce-pr183-codex-2003`
- `git diff --check -- docs/auto-integration.md server/README.md README.md` before metadata commit (clean)
- `git diff --check origin/main..HEAD` after full-stack refresh (reported two pre-existing blank-line-at-EOF warnings in `luce-bench/src/lucebench/fixtures/forge_eval/scenarios/_model_quality.py` and `_stateful_model_quality.py`; not introduced by this run)
- `uv run --directory luce-bench --with pytest pytest -q tests/test_runner.py tests/test_smoke_area.py` (passed)

## Notes

- Primary checkout `/home/erik/Projects/luce2` was clean at start and was not edited directly during reconciliation.
- Retained worktree `/tmp/luce-auto-cron-20260527-1959` for this successful integration refresh until pushed/verified.
- Retained conflicted worktree `/tmp/luce-attempt-pr183-20260527-1959` for audit; it contains unmerged PR #183 probe state.
- Retained prior conflicted worktrees from earlier runs as previously reported; cleanup would be separate maintenance.
