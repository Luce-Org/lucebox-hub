# Dual-Monster Resume Plan — feat/mtp-prefix-warm-ghost

A focused roadmap for the dual-speculator (MTP + DFlash) server, finishing
what we started today and stripping anything that isn't load-bearing.

## Where we are right now (15:48 — server stop, recap)

Committed on `dusterbloom:feat/mtp-prefix-warm-ghost`, pushed:
- `1d0a7d1` mtp: native-heads MTP speculator (Qwen3.6 NextN, γ-chain)
- `4e1a598` mtp: prefix-cache WARM hit
- `230c303` mtp: load NextN head blocks onto GPU by default
- `5e7594c` qwen35: fix do_spec_decode argmax OOB on prefix-cache partial restore
- `af05a23` pflash: propagate skip_park to daemon compress command
- `83e19d9` bench: add 2026-05-19 matrix

Uncommitted on the same branch (live in tree):
- **Option A (dual-speculator switch)** — 272+/82− across 6 files
  (`model_backend.h`, `qwen35_backend.{h,cpp}`, `daemon_loop.cpp`,
  `test_dflash.cpp`, `server.py`). Build clean, smoke-tested
  (dflash/mtp/auto all route correctly, auto threshold = 4096 tokens).
- **bench_matrix.py + matrix/ subpackage** restored from f031f08
  (was never merged to main).
- **PFlash drafter-arch wiring** (+ ~30 lines across `_prefill_hook.py`
  and `qwen35_backend.cpp::handle_compress`) to let the server pass
  `qwen35-0.8b` through to the daemon — added during E1.
- Two new bench result directories with summary.md + JSON artifacts.
- Two thoughts/ documents (architecture + drafter unification plan).
- This file.

Server state: **down** (you stopped it after the dual-spec smoke succeeded).
Last known-good config (3090, 19.9 GB VRAM, dual-loaded):

```
python3 dflash/scripts/server.py --host 127.0.0.1 --port 18080         \
  --target  /home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf \
  --mtp-gguf $TARGET --mtp-gamma 3 --mtp-draft-source chain            \
  --draft   /home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf \
  --bin     dflash/build/test_dflash --budget 22 --verify-mode ddtree  \
  --max-ctx 16384 --fa-window 2048                                     \
  --cache-type-k q8_0 --cache-type-v q8_0
```

## Answered today

1. **Does MTP work end-to-end on Qwen3.6-27B / 3090?**
   YES. Three rebase-time bugs fixed (NextN load, argmax OOB,
   skip_park propagation); MTP decode 50-65 tok/s, accept 0.7-1.0,
   robust to KV quant.

2. **When does MTP beat DFlash?**
   - DFlash wins code/math under ~16K ctx (1.75-2.6× via bench_matrix).
   - **MTP wins agent prompts** (DFlash drafter accept 29% vs MTP 69%).
   - **MTP wins PFlash-compressed long ctx by 2.4-5.7×** (DFlash drafter
     accept collapses to 14-21% on gapped compressed sequences).

3. **Is the hybrid feasible?**
   YES — shipped today as Option A. Backend keeps both speculators
   resident, `GenerateRequest.speculator ∈ {dflash, mtp, auto}` routes
   per request, `auto` defaults to `prompt_tokens > 4096 → MTP`. VRAM
   19.9 GB / 24 GB. Auto-route smoke verified.

4. **No regression vs yesterday's f031f08 matrix.**
   All 9 (3 suites × 3 speculators) within ±5% of f031f08 mean tok/s.

## Open after this — in priority order

### M1 — Commit + push everything that's ready  (~10 min)

```
git add dflash/scripts/bench_matrix.py dflash/scripts/render_matrix.py \
        dflash/scripts/matrix/
git add dflash/src/common/model_backend.h \
        dflash/src/qwen35/qwen35_backend.{h,cpp} \
        dflash/src/common/daemon_loop.cpp \
        dflash/test/test_dflash.cpp \
        dflash/scripts/server.py \
        dflash/scripts/_prefill_hook.py
git add dflash/bench/results/2026-05-19_mtp-prefix-warm-ghost/ \
        dflash/bench/results/2026-05-19T11-54-32_83e19d9/ \
        thoughts/2026-05-19/
```

Three logical commits:
1. `feat(speculator): per-request DFlash/MTP/auto dispatcher (Option A)`
2. `bench: restore bench_matrix orchestrator from f031f08`
3. `docs(thoughts): dual-speculator architecture + unification plan`

Plus PFlash drafter-arch wiring as its own commit:
4. `pflash: pass drafter_arch through compress protocol`

### M2 — Validate Option A on real CLI sessions  (~30-60 min)

Three CLIs to drive against the dual server in turn (hermes, opencode,
pi — your priority order). Tasks:
- Real coding task in this repo, e.g. "Read `dflash/src/qwen35/qwen35_backend.cpp` and explain `do_spec_decode`."
- Expect 5-15 turns of tool-use per CLI.
- Driver script lives at `/tmp/bench-runs/cli_session_driver.py` (ready).
- For each CLI: roll-up per-request server stats, count requests by
  speculator (dflash / mtp / auto-resolved-to-X), accept rates,
  TTFT distribution, decode tok/s distribution, total wall.

Decision artifact: does the 4096-token auto threshold land users on
the right speculator for real workloads, or do they spend too much
time in the wrong regime?

### M3 — Open PR  (~10 min after M1+M2)

Source: `dusterbloom:feat/mtp-prefix-warm-ghost`
Target: `Luce-Org:main`

PR body should call out:
- The three rebase fixes (separate concerns from the feature work)
- Option A as the headline (dual-speculator with `speculator=` field)
- Bench matrix evidence: no kernel regression + agent/long-ctx MTP win
- Known limits: 24 GB minimum, naïve auto threshold, no mid-stream
  switch
- Dependency: PR #195 (draft safetensors loader) — coordinate merge

### M4 — Mid-stream fallback (Option B)  (~1 day)

When `auto` picks DFlash but accept rate over the first N tokens of the
current generation drops below threshold, hot-swap to MTP for the
remainder. Implementation idea: in `do_spec_decode`, after the first 2-3
verify steps, compute running accept-rate; if < 0.35, abort the spec
loop and call `do_mtp_decode_` for the remaining `n_gen - committed`
tokens. Requires the MTP head_kv to be warmable from the current
backbone state — which `warm_mtp_for_prompt_` already does at prefill
boundary, but mid-stream would need a fresh warm.

Defer unless real-CLI data (M2) shows accept regime flips mid-conversation.

### M5 — Smart auto-select with cheap classifier  (~half day)

The current `prompt_tokens > 4096` heuristic is crude. Better options:
- System-prompt fingerprint: "you are a coding agent" / "you are a
  helpful assistant" → MTP. Heuristics extracted once per session.
- Turn-counter: turns 1-3 are usually short → DFlash; turns 4+ usually
  accumulated history → MTP.
- Per-CLI defaults: client-id header → preset.

### M6 — PFlash drafter studies (E1a/E1b in flight)  (~30 min wall)

Just-launched: Qwen3.5-0.8B BF16 + Q4 as PFlash drafter, NIAH 32K-131K.
Already-downloaded models in `/home/peppi/models/qwen3.5-0.8b/`. The
arch wiring (compress protocol + Python `--prefill-drafter-arch` flag)
is uncommitted but functional.

Outcomes feed the unification plan
(`thoughts/2026-05-19/pflash-drafter-unification-plan.md`):
- If Q4-0.8B matches or beats Qwen3-0.6B-BF16 on NIAH recall and
  compress time: ship as the new default PFlash drafter on 3090
  (saves 1 GB VRAM).
- If BF16-0.8B beats both materially: revisit; document the choice.
- If neither moves the needle: keep Qwen3-0.6B-BF16 as canonical and
  close the experiment.

### M7 — Memory-tight profile (16 GB GPUs)  (~half day)

Currently dual-load is hard-coded resident on 24 GB. For 16 GB cards
(4060 Ti, 4080 etc.), need a `--speculator-budget=tight` mode that:
- Parks the unused speculator after each request.
- Re-routes `auto` to whichever is currently loaded.
- Reuses unpark machinery (already exists for DFlash drafter via
  --lazy-draft; MTP unpark is the harder piece — needs the loader fix
  applied to MTP block weights).

Not on the critical path for 3090. Schedule after M3.

### M8 — Stress + sustained throughput  (~half day)

What we haven't tested:
- 100+ concurrent or sequential requests on the same server.
- KV growth over a 1-hour session.
- Drafter memory drift (does compress drafter leak across many calls?).
- Prefix-cache eviction under load.

Schedule after M2 — real-CLI sessions will probably surface the first
failure mode without a dedicated stress run.

## What we explicitly DO NOT plan to ship in this PR

- Path D unification (single drafter for PFlash + DFlash decode) — E1
  surfaced the structural conflict (DFlash drafts strip lm_head, PFlash
  scorer requires it). Documented in `pflash-drafter-unification-plan.md`.
- Mid-stream Option B speculator fallback — defer until data justifies.
- New benches for 192K / 262K NIAH — defer; PFlash compress dominates
  TTFT at those sizes, decode-side architecture is already characterized
  up to 131K.

## Files index (for fast resume)

- Architecture: `thoughts/2026-05-19/dual-speculator-architecture.md`
- PFlash drafter unification: `thoughts/2026-05-19/pflash-drafter-unification-plan.md`
- This resume plan: `thoughts/2026-05-19/dual-monster-resume-plan.md`
- Bench summary (today): `dflash/bench/results/2026-05-19_mtp-prefix-warm-ghost/summary.md`
- Bench matrix today: `dflash/bench/results/2026-05-19T11-54-32_83e19d9/summary.md`
- CLI driver (ready): `/tmp/bench-runs/cli_session_driver.py`
- NIAH sweep script: `/tmp/bench-runs/niah_sweep.py`
