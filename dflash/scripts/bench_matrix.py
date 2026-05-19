"""bench_matrix.py — workload × speculator benchmarking orchestrator.

Usage:
    python3 bench_matrix.py \\
        --workloads swe_bench_2k,swe_bench_8k \\
        --speculators ar,mtp_d3,dflash_b22 \\
        --n-gen 128 \\
        --n-runs 3 \\
        --out-dir dflash/bench/results

Each run produces a versioned artifact directory:
    <out-dir>/<timestamp>_<git-sha>/
        meta.json
        <workload>_x_<speculator>.json   (one per pair)
        summary.md

The AR speculator result is cached per workload run and reused for
speedup computation in every other speculator's artifact — no redundant
AR re-runs.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Ensure matrix package is importable from the same scripts/ dir.
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from matrix.speculator import SpeculatorResult
from matrix.workload import WorkloadPrompt

# ── speculator registry ────────────────────────────────────────────────────

def _build_speculator(name: str):
    """Instantiate a speculator by short name.

    Supported names:
        ar           — ARSpeculator (gamma=0)
        mtp_dN       — MTPSpeculator(chain_depth=N)
        dflash_bN    — DFlashSpeculator(budget=N)
    """
    from matrix.speculators.ar import ARSpeculator
    from matrix.speculators.mtp import MTPSpeculator
    from matrix.speculators.dflash import DFlashSpeculator

    if name == "ar":
        return ARSpeculator()
    if name.startswith("mtp_d"):
        depth = int(name[len("mtp_d"):])
        return MTPSpeculator(chain_depth=depth)
    if name.startswith("dflash_b"):
        budget = int(name[len("dflash_b"):])
        return DFlashSpeculator(budget=budget)
    raise ValueError(
        f"Unknown speculator {name!r}. "
        "Valid forms: ar | mtp_dN | dflash_bN"
    )


# ── workload registry ──────────────────────────────────────────────────────

def _build_workload(name: str, n_sample: int, seed: int):
    """Instantiate a workload by short name.

    Supported names:
        swe_bench_2k / swe_bench_8k / swe_bench_24k
        humaneval   — openai_humaneval test split
        gsm8k       — gsm8k main test split
        math500     — HuggingFaceH4/MATH-500 test split
    """
    from matrix.workloads.swe_bench import SweBenchWorkload
    from matrix.workloads.humaneval import HumanEvalWorkload
    from matrix.workloads.gsm8k import Gsm8kWorkload
    from matrix.workloads.math500 import Math500Workload

    if name.startswith("swe_bench_"):
        bucket = name[len("swe_bench_"):]
        return SweBenchWorkload(bucket=bucket, n_sample=n_sample, seed=seed)
    if name == "humaneval":
        return HumanEvalWorkload(n_sample=n_sample, seed=seed)
    if name == "gsm8k":
        return Gsm8kWorkload(n_sample=n_sample, seed=seed)
    if name == "math500":
        return Math500Workload(n_sample=n_sample, seed=seed)
    raise ValueError(
        f"Unknown workload {name!r}. "
        "Valid forms: swe_bench_2k | swe_bench_8k | swe_bench_24k | "
        "humaneval | gsm8k | math500"
    )


# ── hardware / commit capture ──────────────────────────────────────────────

def _capture_meta(schema_v: int = 1) -> Dict[str, Any]:
    """Capture run-level metadata."""
    ts = datetime.now(timezone.utc).isoformat()

    git_sha = "unknown"
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(SCRIPTS.parent),
        )
        if r.returncode == 0:
            git_sha = r.stdout.strip()
    except Exception:
        pass

    gpu_info = "unknown"
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            gpu_info = r.stdout.strip()
    except Exception:
        pass

    cuda_version = "unknown"
    try:
        r = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            cuda_version = r.stdout.strip().splitlines()[-1]
    except Exception:
        pass

    return {
        "schema_v": schema_v,
        "timestamp_utc": ts,
        "git_sha": git_sha,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu": gpu_info,
        "cuda": cuda_version,
    }


# ── bootstrap CI ──────────────────────────────────────────────────────────

def _bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI.  Returns (ci_low, ci_high)."""
    rng = random.Random(seed)
    n = len(values)
    if n < 2:
        v = values[0] if values else 0.0
        return v, v
    boot_medians = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_medians.append(statistics.median(sample))
    boot_medians.sort()
    lo = boot_medians[int(0.025 * n_resamples)]
    hi = boot_medians[int(0.975 * n_resamples)]
    return lo, hi


# ── aggregation ────────────────────────────────────────────────────────────

def _aggregate(
    results: List[Dict[str, Any]],
    ar_results: List[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    """Compute aggregate statistics over a list of per-prompt result dicts."""
    tok_s_vals = [r["tok_s"] for r in results]
    mean_val = statistics.mean(tok_s_vals) if tok_s_vals else 0.0
    median_val = statistics.median(tok_s_vals) if tok_s_vals else 0.0
    p25 = statistics.quantiles(tok_s_vals, n=4)[0] if len(tok_s_vals) >= 2 else median_val
    p75 = statistics.quantiles(tok_s_vals, n=4)[2] if len(tok_s_vals) >= 2 else median_val
    ci_lo, ci_hi = _bootstrap_ci(tok_s_vals)

    agg: Dict[str, Any] = {
        "tok_s_mean": mean_val,
        "tok_s_median": median_val,
        "tok_s_p25": p25,
        "tok_s_p75": p75,
        "tok_s_ci95_low": ci_lo,
        "tok_s_ci95_high": ci_hi,
        "n_total_runs": len(results),
    }

    if ar_results:
        ar_vals = [r["tok_s"] for r in ar_results]
        ar_median = statistics.median(ar_vals) if ar_vals else None
        ar_mean = statistics.mean(ar_vals) if ar_vals else None
        if ar_median and ar_median > 0:
            agg["spec_vs_ar_speedup_median"] = round(median_val / ar_median, 4)
        if ar_mean and ar_mean > 0:
            agg["spec_vs_ar_speedup_mean"] = round(mean_val / ar_mean, 4)

    return agg


# ── result serialisation ───────────────────────────────────────────────────

def _result_to_dict(
    idx: int,
    prompt: WorkloadPrompt,
    res: SpeculatorResult,
) -> Dict[str, Any]:
    return {
        "prompt_idx": idx,
        "prompt_id": prompt.prompt_id,
        "tok_s": res.tok_s,
        "accept_rate": res.accept_rate,
        "al": res.al,
        "n_tokens_generated": res.n_tokens_generated,
        "decode_s": res.decode_s,
        "prefill_s": res.prefill_s,
        "raw_json": res.raw_json,
    }


# ── per-(workload, speculator) run ─────────────────────────────────────────

def run_pair(
    workload,
    speculator,
    n_gen: int,
    n_runs: int,
    ar_cache: Dict[str, List[Dict[str, Any]]] | None,
    tmpdir: Path,
) -> Tuple[List[Dict[str, Any]], List[WorkloadPrompt]]:
    """Run one (workload, speculator) pair.

    Returns (flat_results, prompts_used) where each entry in flat_results
    corresponds to one (prompt, run) combination.
    """
    results: List[Dict[str, Any]] = []
    prompts_used: List[WorkloadPrompt] = []

    prompts = list(workload.prompts())

    for run_idx in range(n_runs):
        print(
            f"  [run {run_idx + 1}/{n_runs}] {workload.name} × {speculator.name}",
            flush=True,
        )
        for prompt in prompts:
            bin_path = getattr(prompt, "bin_path", None)
            if bin_path is None:
                bin_path = prompt.write_bin(
                    tmpdir / f"{workload.name}_{prompt.idx:03d}.bin"
                )

            max_ctx = speculator._max_ctx_for(prompt.n_prompt_tokens, n_gen)
            try:
                res = speculator.run(
                    prompt_path=str(bin_path),
                    prompt_id=prompt.idx,
                    n_prompt=prompt.n_prompt_tokens,
                    n_gen=n_gen,
                    max_ctx=max_ctx,
                )
            except Exception as exc:
                print(
                    f"    WARN: {speculator.name} prompt={prompt.prompt_id} "
                    f"run={run_idx + 1} FAILED: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            row = _result_to_dict(prompt.idx, prompt, res)
            row["run_idx"] = run_idx
            results.append(row)
            if run_idx == 0:
                prompts_used.append(prompt)

            print(
                f"    prompt={prompt.prompt_id}  tok_s={res.tok_s:.2f}  "
                f"accept={res.accept_rate if res.accept_rate is not None else 'n/a'}  "
                f"prefill={res.prefill_s:.2f}s  decode={res.decode_s:.2f}s",
                flush=True,
            )

    return results, prompts_used


# ── artifact writing ───────────────────────────────────────────────────────

def _write_artifact(
    out_dir: Path,
    workload,
    speculator,
    run_meta: Dict[str, Any],
    prompts: List[WorkloadPrompt],
    results: List[Dict[str, Any]],
    ar_results: List[Dict[str, Any]] | None,
) -> Path:
    prompt_shas = [p.sha256 for p in prompts]
    agg = _aggregate(results, ar_results)

    artifact = {
        "meta": run_meta,
        "workload": {
            **workload.config(),
            "prompt_shas": prompt_shas,
        },
        "speculator": {
            "name": speculator.name,
            "config": speculator.config(),
        },
        "results": results,
        "aggregates": agg,
    }

    fname = f"{workload.name}_x_{speculator.name}.json"
    path = out_dir / fname
    path.write_text(json.dumps(artifact, indent=2, default=str))
    print(f"  Wrote {path}", flush=True)
    return path


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="bench_matrix: workload × speculator benchmarking orchestrator",
    )
    parser.add_argument(
        "--workloads",
        default="swe_bench_2k",
        help="Comma-separated workload names: swe_bench_2k,swe_bench_8k,humaneval,gsm8k,math500",
    )
    parser.add_argument(
        "--speculators",
        default="ar,mtp_d3,dflash_b22",
        help="Comma-separated speculator names: ar,mtp_d3,dflash_b22",
    )
    parser.add_argument(
        "--n-gen", type=int, default=128,
        help="Tokens to generate per prompt (default 128)",
    )
    parser.add_argument(
        "--n-runs", type=int, default=3,
        help="Runs per (workload, speculator) pair (default 3)",
    )
    parser.add_argument(
        "--n-sample", type=int, default=8,
        help="Prompts per workload (default 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for workload sampling (default 42)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dflash/bench/results"),
        help="Output root directory (default dflash/bench/results)",
    )
    parser.add_argument(
        "--no-ar-cache",
        action="store_true",
        help="Disable AR result caching (re-run AR for every speculator pair)",
    )
    args = parser.parse_args()

    workload_names = [w.strip() for w in args.workloads.split(",") if w.strip()]
    speculator_names = [s.strip() for s in args.speculators.split(",") if s.strip()]

    print(f"Workloads:   {workload_names}", flush=True)
    print(f"Speculators: {speculator_names}", flush=True)
    print(f"n_runs={args.n_runs}  n_gen={args.n_gen}  n_sample={args.n_sample}", flush=True)

    meta = _capture_meta()
    sha_short = meta["git_sha"][:7] if meta["git_sha"] != "unknown" else "unknown"
    ts_tag = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = args.out_dir / f"{ts_tag}_{sha_short}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}", flush=True)

    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    tmpdir = Path(tempfile.gettempdir()) / "dflash_bench_matrix"
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Cache AR results per workload name so we only run AR once.
    ar_cache: Dict[str, List[Dict[str, Any]]] = {}
    ar_prompts_cache: Dict[str, List[WorkloadPrompt]] = {}

    artifact_paths: List[Path] = []

    for wname in workload_names:
        try:
            workload = _build_workload(wname, n_sample=args.n_sample, seed=args.seed)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        # Run AR first (or reuse cache) so speedup can be computed for all specs.
        if not args.no_ar_cache and wname not in ar_cache:
            print(f"\n=== AR baseline for {wname} ===", flush=True)
            ar_spec = _build_speculator("ar")
            ar_workload = _build_workload(wname, n_sample=args.n_sample, seed=args.seed)
            ar_results, ar_prompts = run_pair(
                ar_workload, ar_spec, args.n_gen, args.n_runs,
                ar_cache=None, tmpdir=tmpdir,
            )
            ar_cache[wname] = ar_results
            ar_prompts_cache[wname] = ar_prompts
            ap = _write_artifact(
                run_dir, ar_workload, ar_spec, meta,
                ar_prompts, ar_results, ar_results=None,
            )
            artifact_paths.append(ap)

        for sname in speculator_names:
            if sname == "ar" and not args.no_ar_cache and wname in ar_cache:
                # Already ran AR above — skip re-run.
                continue
            print(f"\n=== {wname} × {sname} ===", flush=True)
            try:
                spec = _build_speculator(sname)
            except ValueError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                continue

            wl = _build_workload(wname, n_sample=args.n_sample, seed=args.seed)
            results, prompts = run_pair(
                wl, spec, args.n_gen, args.n_runs,
                ar_cache=ar_cache.get(wname),
                tmpdir=tmpdir,
            )
            ar_res = ar_cache.get(wname) if not args.no_ar_cache else None
            ap = _write_artifact(
                run_dir, wl, spec, meta,
                ar_prompts_cache.get(wname, prompts), results, ar_res,
            )
            artifact_paths.append(ap)

    # Render summary markdown.
    import importlib.util
    render_script = SCRIPTS / "render_matrix.py"
    if render_script.exists():
        spec_render = importlib.util.spec_from_file_location("render_matrix", render_script)
        render_mod = importlib.util.module_from_spec(spec_render)  # type: ignore
        spec_render.loader.exec_module(render_mod)  # type: ignore[union-attr]
        summary_md = render_mod.render(run_dir)
        (run_dir / "summary.md").write_text(summary_md)
        print(f"\nSummary: {run_dir / 'summary.md'}", flush=True)

    print(f"\nDone. Artifacts in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
