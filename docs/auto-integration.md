# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T11:55:49-04:00
Current base: `origin/main` `fda8877b`
Current integration tip before this refresh: `easel/auto-integration` `e8548491`
Refreshed stack tip prepared in this run: this commit

This branch is maintained as a reproducible patch stack over `origin/main`.
The primary checkout was clean at the start of this unattended run. This refresh
rebases the existing stack over the latest upstream `origin/main`, preserves all
currently integrated non-draft contributor PR refs, records fresh direct merge
probes for the remaining non-ancestor PRs, and adds one integration-only include
fix needed by the new upstream bounded feature-range restore path.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #292 | `feat-backend-ipc-payload-pipe-open` | `90bc52f` | included | Adds backend IPC payload-pipe support for remote draft feature/noise payloads, with integration-only pipe-drain hardening and shared feature-slice storage helper. |
| #289 | `pipeline_moe` | `0ffab8a` | included | Pipelined hybrid Qwen35 MoE decode update is carried. The inaccessible submodule pointer from earlier PR history remains excluded. |
| #284 | `fix/draft-safetensors-rope-theta` | `63bba30` | included | Reads and validates `rope_theta` from draft safetensors `config.json`. |
| #278 | `fix-pflash-drafter-backend-precision-submit` | `fdfcbda` | included | Adds legacy CUDA drafter precision fallback via shared backend precision policy while preserving current Q8_0 allocation behavior where present. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `0e3c79a` | included | Qwen3.6-27B tool-calling fix for Claude-code Anthropic path. |
| #274 | `feat/pflash-drafter-ee7` | `2c19f66` | included | Adaptive pFlash compression docs/config, transitive anchor default, backup-suffix ignores, EE7/drafter updates, and the Qwen3 closed-`<think>` Jinja prefill docs/tests from the branch's latest head are carried. |
| #266 | `feat/harness-typed-adapters` | `17525ea` | included | Typed harness adapters and format-aware session-inject proxy. |
| #265 | `feat-cpp-server-target-layer-split-prep` | upstream | included through upstream | Upstream main contains this PR. |
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
| upstream sync | integrated | `origin/main` advanced from `43457d8f` to `fda8877b`. Reconciliation worktree `/tmp/luce-auto-cron-20260528-115635` started from `easel/auto-integration` `e8548491` and merged upstream as `4da4424f`. |
| integration-only fix | applied | Added missing `<fstream>` include in `server/src/common/dflash_draft_ipc_daemon.cpp`; the new upstream bounded `set_feature_range` restore path uses `std::ifstream`. |
| current integrated PRs | checked | After upstream sync, `git merge-base --is-ancestor origin/pr/<n> HEAD` shows #292, #289, #284, #278, #276, #274, #266, #152, and #142 are ancestors of the refreshed stack. #265 is included through `origin/main`. |
| remaining non-ancestor non-draft PRs | direct merge probes still conflicted | Fresh isolated probes attempted `--no-commit --no-ff` merges for #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39 against the post-upstream-sync stack. Every direct probe conflicted and was aborted in the isolated probe worktree. Consolidated output: `/tmp/luce-merge-probes-20260528-115635.txt`. |
| #237 delegated review carry-forward | still applicable | Prior tmux-driven Claude/Codex attempts remain the latest substantial delegated feasibility work for the MTP-foundation blocker: Claude failed to produce a usable report; Codex reported that direct merge is unsafe due to the `dflash/src` to `server/src` relocation and recommended a deliberate current-layout MTP architecture port. |

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Current status | Next useful action |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Direct merge still conflicts across current server/common/qwen35 files and old `dflash/` paths. Fresh probes confirm this remains architectural port work: first port common MTP interfaces/runners and tests under `server/`, then add MTP fields to `BackendArgs`/`ModelBackend`, then reauthor Qwen35 hidden/logit/rollback capture against current `StepGraph`/MoE changes, then wire `qwen35_mtp*`, CLI flags, CMake, and focused tests. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Requires #237/equivalent MTP foundation, then a current-layout feature port for hidden-state/speculator hooks, backend dispatcher, daemon wiring, and WARM tests. |
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

- `date -Is` -> 2026-05-28T11:55:49-04:00 at preflight.
- Primary checkout `git status --short` was clean before work began.
- `git branch --show-current` reported `auto-integration` in the primary checkout.
- `git remote -v` verified `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status` succeeded for account `easel` with repo/workflow scopes.
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text` succeeded for the Claude Team account.
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version` succeeded (`codex-cli 0.130.0`).
- `git fetch --prune origin` and `git fetch --prune easel` completed; targeted fetches recreated current open non-draft contributor PR refs.
- Isolated reconciliation worktree `/tmp/luce-auto-cron-20260528-115635` merged latest `origin/main` (`fda8877b`) into the stack as `4da4424f`.
- Isolated probe worktree `/tmp/luce-probe-20260528-115635` reran direct conflict probes for all remaining non-ancestor non-draft PRs.
- Ancestor checks passed for included contributor PR refs #292, #289, #284, #278, #276, #274, #266, #152, and #142.
- `git diff --check` passed for the upstream-sync merge, integration-only include fix, and manifest refresh.
- Targeted conflict-marker scan over `server/src/common/dflash_draft_ipc_daemon.cpp`, `server/src/common/layer_split_utils.cpp`, and `docs/auto-integration.md` found no merge markers.
- A direct `g++ -fsyntax-only` probe for `server/src/common/dflash_draft_ipc_daemon.cpp` could not compile because the worktree's submodules are not initialized (`ggml.h` absent under `server/deps/llama.cpp`).
- `cmake -S server -B /tmp/luce-cmake-20260528-115635 -DDFLASH27B_GPU_BACKEND=cuda -DCMAKE_CUDA_ARCHITECTURES=89` failed during CUDA compiler identification because local `/usr/bin/nvcc`/CMake still selected unsupported `sm_52` (`ptxas fatal : Value 'sm_52' is not defined`), before project compilation.

## Notes

- Primary checkout `/home/erik/Projects/luce2` was clean at preflight and matched fetched `easel/auto-integration` (`e8548491`).
- Retained reconciliation worktree `/tmp/luce-auto-cron-20260528-115635` for audit/final push preparation.
- Retained probe worktree `/tmp/luce-probe-20260528-115635` plus direct-probe log `/tmp/luce-merge-probes-20260528-115635.txt`.
- Prior retained worktrees, probe logs, agent reports, and configure directories remain as listed in earlier manifest revisions; cleanup is separate maintenance.
