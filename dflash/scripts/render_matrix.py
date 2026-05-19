"""render_matrix.py — produce a markdown comparison table from bench_matrix artifacts.

Usage:
    python3 render_matrix.py dflash/bench/results/2026-05-17T13-15-00_abc1234/

Produces a markdown table to stdout (and writes summary.md in the run dir if
called as a script).  Also callable as a library: render(run_dir) -> str.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple


# ── ASCII histogram (5 bins) ───────────────────────────────────────────────

def _ascii_hist(values: List[float], n_bins: int = 5, width: int = 10) -> str:
    """Return a one-line ASCII histogram string for a list of floats."""
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return f"[{'|' * width}] all={lo:.1f}"
    step = (hi - lo) / n_bins
    counts = [0] * n_bins
    for v in values:
        b = min(n_bins - 1, int((v - lo) / step))
        counts[b] += 1
    max_count = max(counts) or 1
    bars = []
    for c in counts:
        bar_len = max(1, round(c / max_count * width)) if c > 0 else 0
        bars.append("█" * bar_len if bar_len else " ")
    return f"[{'|'.join(bars)}] {lo:.0f}–{hi:.0f}"


# ── artifact loading ───────────────────────────────────────────────────────

def _load_artifacts(run_dir: Path) -> List[Dict[str, Any]]:
    """Load all *_x_*.json artifacts from a run directory."""
    arts = []
    for p in sorted(run_dir.glob("*_x_*.json")):
        try:
            arts.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"WARN: could not load {p}: {e}", file=sys.stderr)
    return arts


# ── main render function ───────────────────────────────────────────────────

def render(run_dir: Path) -> str:
    """Render a markdown comparison table for all artifacts in run_dir."""
    run_dir = Path(run_dir)
    arts = _load_artifacts(run_dir)
    if not arts:
        return f"# No artifacts found in {run_dir}\n"

    # Read run-level meta.
    meta_path = run_dir / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass

    git_sha = meta.get("git_sha", "unknown")[:7]
    gpu = meta.get("gpu", "unknown")
    platform_s = meta.get("platform", "unknown")
    cuda = meta.get("cuda", "unknown")
    ts = meta.get("timestamp_utc", "")[:10]

    # Group artifacts by workload then speculator.
    # Structure: {workload_name: {speculator_name: artifact}}
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for art in arts:
        wname = art.get("workload", {}).get("name", "unknown")
        sname = art.get("speculator", {}).get("name", "unknown")
        grouped.setdefault(wname, {})[sname] = art

    # Determine column order: ar first, then alphabetical.
    all_specs: List[str] = []
    for specs in grouped.values():
        for s in specs:
            if s not in all_specs:
                all_specs.append(s)
    all_specs.sort(key=lambda s: (0 if s == "ar" else 1, s))

    lines: List[str] = []
    lines.append(f"# Bench matrix — {ts}  commit {git_sha}")
    lines.append("")
    lines.append(f"GPU: {gpu}  |  {platform_s}  |  {cuda}")
    lines.append("")

    for wname, specs in sorted(grouped.items()):
        # Use any speculator's metadata (the first one with non-empty data),
        # not necessarily the first by alphabetical order — AR may have
        # failed even when others succeeded.
        sample_spec = next(
            (s for s in specs.values() if s.get("aggregates", {}).get("n_total_runs", 0) > 0),
            specs.get(all_specs[0] if all_specs else "ar", {}),
        )
        n_sample = sample_spec.get("workload", {}).get("n_sample", "?")
        n_runs = sample_spec.get("aggregates", {}).get("n_total_runs", "?")
        lines.append(f"## {wname}  (n_sample={n_sample}, n_runs={n_runs})")
        lines.append("")

        # Build table header.
        header_cols = ["speculator", "tok/s median", "tok/s p25–p75", "CI 95%",
                       "prefill s med", "AL/accept", "speedup vs AR", "distribution"]
        sep = ["-" * len(c) for c in header_cols]
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("| " + " | ".join(sep) + " |")

        ar_agg = specs.get("ar", {}).get("aggregates", {})
        ar_median = ar_agg.get("tok_s_median", None)

        for sname in all_specs:
            if sname not in specs:
                continue
            art = specs[sname]
            agg = art.get("aggregates", {})
            tok_med = agg.get("tok_s_median", None)
            p25 = agg.get("tok_s_p25", None)
            p75 = agg.get("tok_s_p75", None)
            ci_lo = agg.get("tok_s_ci95_low", None)
            ci_hi = agg.get("tok_s_ci95_high", None)

            # Per-result aggregates.
            results_list = art.get("results", [])
            ar_vals = [r["accept_rate"] for r in results_list
                       if r.get("accept_rate") is not None]
            al_vals = [r["al"] for r in results_list if r.get("al") is not None]
            prefill_vals = [r["prefill_s"] for r in results_list
                            if r.get("prefill_s") is not None]

            # AL/accept: prefer AL when present (DFlash), else accept_rate (MTP).
            if al_vals:
                al_med = statistics.median(al_vals)
                al_str = f"AL={al_med:.2f}"
            elif ar_vals:
                al_str = f"acc={statistics.mean(ar_vals) * 100:.1f}%"
            else:
                al_str = "n/a"

            prefill_med = (
                statistics.median(prefill_vals) if prefill_vals else None
            )

            # Speedup vs AR.
            speedup = agg.get("spec_vs_ar_speedup_median")
            if speedup is None and ar_median and tok_med:
                speedup = tok_med / ar_median
            speedup_str = f"{speedup:.2f}x" if speedup is not None else "—"

            tok_vals = [r["tok_s"] for r in results_list]

            def _fmt(v):
                return f"{v:.2f}" if v is not None else "—"

            row = [
                sname,
                _fmt(tok_med),
                f"{_fmt(p25)}–{_fmt(p75)}",
                f"{_fmt(ci_lo)}–{_fmt(ci_hi)}",
                _fmt(prefill_med),
                al_str,
                speedup_str,
                _ascii_hist(tok_vals),
            ]
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: render_matrix.py <run-dir>", file=sys.stderr)
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    md = render(run_dir)
    print(md)
    out = run_dir / "summary.md"
    out.write_text(md)
    print(f"\nWrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
