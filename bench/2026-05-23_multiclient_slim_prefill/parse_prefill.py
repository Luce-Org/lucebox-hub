#!/usr/bin/env python3
"""Parse server logs and run CSVs to produce the prefill capture CSV format.

For each client+condition, parses per-turn:
- prompt_tokens_in: from [pflash] N -> ... OR prompt_tokens from [prefill] tokens=N
- prompt_tokens_after_pflash: from [pflash] ... -> K tokens
- prefill_time_logged_s: from [prefill] time=X.XXX s
- decode_time_s: from [spec-decode] time=Y.YYY s
- output_tokens: from [spec-decode] tokens=N
- accept_rate: from [spec-decode] accepted=M/K (X%)
- wall_s: from run CSV

Also computes prefill_time_est_s = wall_s - decode_time_s (rough estimate).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

OUTDIR = Path(__file__).parent
CLIENTS = ["claude_code", "hermes", "codex", "pi", "opencode"]
CONDITIONS = ["baseline", "slim"]

# Regex patterns for server log parsing
RE_PFLASH = re.compile(r'\[pflash\]\s+(\d+)\s+->\s+\d+\s+->\s+(\d+)\s+tokens')
RE_PREFILL = re.compile(r'\[prefill\]\s+tokens=(\d+)\s+time=([\d.]+)\s+s')
RE_SPECDEC = re.compile(r'\[spec-decode\]\s+tokens=(\d+)\s+time=([\d.]+)\s+s.*?accepted=(\d+)/(\d+)\s+\(([\d.]+)%\)')

CSV_COLUMNS = [
    "client", "condition", "turn", "session_id", "prompt_name",
    "prompt_tokens_in", "prompt_tokens_after_pflash",
    "prefill_time_logged_s", "prefill_time_est_s",
    "decode_time_s", "output_tokens", "accept_rate", "wall_s", "ok_done",
]


def parse_server_log(log_text: str) -> list[dict]:
    """Parse server log and return per-request records.

    Log order per request: [pflash] -> [target loaded/snap] -> [spec-decode] -> [prefill].
    [prefill] is logged after generate() returns; [spec-decode] is inside generate().
    So we look BACKWARD from [prefill] for both [pflash] and [spec-decode].
    """
    lines = log_text.splitlines()
    records = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for [pflash] line (compression active)
        pflash_m = RE_PFLASH.search(line)
        prefill_m = RE_PREFILL.search(line)

        if prefill_m:
            tokens_in = int(prefill_m.group(1))
            prefill_time = float(prefill_m.group(2))
            tokens_after = tokens_in  # no pflash unless we saw a pflash line just before
            tokens_raw_in = tokens_in

            # Look back up to 30 lines for [pflash] and [spec-decode] lines.
            # On cold turn 1, many lines (model load, snap alloc) appear between pflash and prefill.
            decode_time = None
            output_tokens = None
            accept_rate = None
            for back in range(1, 31):
                if i - back < 0:
                    break
                prev = lines[i - back]
                # Stop looking back once we hit a previous [prefill] line
                if RE_PREFILL.search(prev) and back > 0:
                    break
                pm = RE_PFLASH.search(prev)
                if pm and tokens_raw_in == tokens_in:
                    tokens_raw_in = int(pm.group(1))
                    tokens_after = int(pm.group(2))
                sdm = RE_SPECDEC.search(prev)
                if sdm and decode_time is None:
                    output_tokens = int(sdm.group(1))
                    decode_time = float(sdm.group(2))
                    accepted = int(sdm.group(3))
                    total = int(sdm.group(4))
                    accept_rate = round(accepted / total, 4) if total > 0 else None

            records.append({
                "prompt_tokens_in": tokens_raw_in,
                "prompt_tokens_after_pflash": tokens_after,
                "prefill_time_logged_s": prefill_time,
                "decode_time_s": decode_time,
                "output_tokens": output_tokens,
                "accept_rate": accept_rate,
            })

        elif pflash_m and not prefill_m:
            pass  # handled via look-back above

        i += 1
    return records


def load_run_csv(csv_path: Path) -> list[dict]:
    """Load the bandit-session CSV (per-turn wall_s etc)."""
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def process_client(client: str, condition: str) -> list[dict]:
    cond_dir = OUTDIR / condition
    server_log_path = cond_dir / f"{client}_server.log"
    run_csv_path = cond_dir / f"{client}.csv"

    if not server_log_path.exists():
        print(f"  SKIP: {client}/{condition} — no server log", file=sys.stderr)
        return []

    log_text = server_log_path.read_text(errors="replace")
    server_records = parse_server_log(log_text)
    run_rows = load_run_csv(run_csv_path)

    # Match server records to turns (server may have more records due to billing/tool calls)
    # Heuristic: group records by large gaps or just use sequential assignment
    # For 3-turn runs, take the last 3 main-prompt records (skip tiny <100-token records)
    main_records = [r for r in server_records if r["prompt_tokens_in"] > 100]

    result_rows = []
    n_turns = max(len(run_rows), 1) if run_rows else 3

    for turn_idx in range(min(n_turns, len(main_records))):
        srv = main_records[turn_idx] if turn_idx < len(main_records) else {}
        run = run_rows[turn_idx] if turn_idx < len(run_rows) else {}

        wall_s = float(run.get("wall_s") or 0) if run else None
        decode_s = srv.get("decode_time_s")
        prefill_logged = srv.get("prefill_time_logged_s")
        prefill_est = round(wall_s - decode_s, 3) if (wall_s and decode_s) else None

        row = {
            "client": client,
            "condition": condition,
            "turn": turn_idx + 1,
            "session_id": run.get("session_id", "") if run else "",
            "prompt_name": run.get("prompt", "") if run else "",
            "prompt_tokens_in": srv.get("prompt_tokens_in", ""),
            "prompt_tokens_after_pflash": srv.get("prompt_tokens_after_pflash", ""),
            "prefill_time_logged_s": prefill_logged,
            "prefill_time_est_s": prefill_est,
            "decode_time_s": decode_s,
            "output_tokens": srv.get("output_tokens", ""),
            "accept_rate": srv.get("accept_rate", ""),
            "wall_s": wall_s,
            "ok_done": "",  # not captured in current harness
        }
        result_rows.append(row)

    return result_rows


def compute_summary(all_rows: list[dict]) -> dict:
    """Compute per-client baseline vs slim deltas."""
    from collections import defaultdict
    by_client_cond: dict[tuple, list[dict]] = defaultdict(list)
    for row in all_rows:
        key = (row["client"], row["condition"])
        by_client_cond[key].append(row)

    clients = sorted({r["client"] for r in all_rows})
    summary = {}
    for client in clients:
        baseline_rows = by_client_cond.get((client, "baseline"), [])
        slim_rows = by_client_cond.get((client, "slim"), [])

        def turn_prefill(rows, turn):
            for r in rows:
                if r["turn"] == turn:
                    v = r.get("prefill_time_logged_s")
                    return float(v) if v is not None else None
            return None

        def avg_accept(rows):
            vals = [float(r["accept_rate"]) for r in rows if r.get("accept_rate")]
            return round(sum(vals) / len(vals), 4) if vals else None

        def total_wall(rows):
            vals = [float(r["wall_s"]) for r in rows if r.get("wall_s")]
            return round(sum(vals), 3) if vals else None

        summary[client] = {
            "baseline": {
                "turn1_prefill_s": turn_prefill(baseline_rows, 1),
                "turn2_prefill_s": turn_prefill(baseline_rows, 2),
                "turn3_prefill_s": turn_prefill(baseline_rows, 3),
                "avg_accept_rate": avg_accept(baseline_rows),
                "total_wall_s": total_wall(baseline_rows),
            },
            "slim": {
                "turn1_prefill_s": turn_prefill(slim_rows, 1),
                "turn2_prefill_s": turn_prefill(slim_rows, 2),
                "turn3_prefill_s": turn_prefill(slim_rows, 3),
                "avg_accept_rate": avg_accept(slim_rows),
                "total_wall_s": total_wall(slim_rows),
            },
        }

        # Verdict
        b = summary[client]["baseline"]
        s = summary[client]["slim"]
        verdicts = []
        # prefill: slim <= baseline is PASS
        for turn in [1, 2, 3]:
            bp = b.get(f"turn{turn}_prefill_s")
            sp = s.get(f"turn{turn}_prefill_s")
            if bp is not None and sp is not None:
                if sp > bp * 1.05:  # allow 5% slack
                    verdicts.append(f"FAIL: turn{turn} prefill {sp:.3f}s > baseline {bp:.3f}s")

        # accept_rate: slim should not be more than 2pp WORSE than baseline
        if b["avg_accept_rate"] and s["avg_accept_rate"]:
            delta = s["avg_accept_rate"] - b["avg_accept_rate"]
            if delta < -0.02:  # only fail if slim is worse by >2pp
                verdicts.append(f"FAIL: accept_rate regression {delta:.4f} (slim < baseline - 0.02)")

        summary[client]["verdict"] = "PASS" if not verdicts else "FAIL"
        summary[client]["verdict_detail"] = verdicts

    return summary


def main():
    all_rows = []
    for condition in CONDITIONS:
        for client in CLIENTS:
            rows = process_client(client, condition)
            all_rows.extend(rows)
            print(f"  {client}/{condition}: {len(rows)} rows")

    # Write combined CSV
    combined_csv = OUTDIR / "all_turns.csv"
    with open(combined_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"\nWrote {len(all_rows)} rows to {combined_csv}")

    # Write per-client per-condition CSVs
    from collections import defaultdict
    by_client_cond: dict[tuple, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_client_cond[(row["client"], row["condition"])].append(row)

    for (client, condition), rows in by_client_cond.items():
        path = OUTDIR / condition / f"{client}_prefill.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
            w.writeheader()
            for row in rows:
                w.writerow(row)

    # Compute summary
    summary = compute_summary(all_rows)

    summary_path = OUTDIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {summary_path}")

    # Print table
    print("\n=== Per-client prefill comparison ===")
    print(f"{'Client':12s}  {'Cond':8s}  {'T1 pf':8s}  {'T2 pf':8s}  {'T3 pf':8s}  {'accept':6s}  {'wall':7s}  {'verdict'}")
    for client in CLIENTS:
        if client not in summary:
            continue
        for cond in CONDITIONS:
            d = summary[client].get(cond, {})
            t1 = f"{d.get('turn1_prefill_s', 0):.3f}" if d.get('turn1_prefill_s') else "n/a"
            t2 = f"{d.get('turn2_prefill_s', 0):.3f}" if d.get('turn2_prefill_s') else "n/a"
            t3 = f"{d.get('turn3_prefill_s', 0):.3f}" if d.get('turn3_prefill_s') else "n/a"
            ar = f"{d.get('avg_accept_rate', 0):.3f}" if d.get('avg_accept_rate') else "n/a"
            wall = f"{d.get('total_wall_s', 0):.1f}" if d.get('total_wall_s') else "n/a"
            verdict = summary[client].get("verdict", "-") if cond == "slim" else ""
            print(f"{client:12s}  {cond:8s}  {t1:8s}  {t2:8s}  {t3:8s}  {ar:6s}  {wall:7s}  {verdict}")

    # Headline
    slims = [(c, summary[c]["slim"]["turn1_prefill_s"], summary[c]["baseline"]["turn1_prefill_s"])
             for c in CLIENTS if c in summary
             and summary[c]["slim"].get("turn1_prefill_s") and summary[c]["baseline"].get("turn1_prefill_s")]
    if slims:
        ratios = [b / s if s > 0 else 1.0 for _, s, b in slims]
        avg_ratio = sum(ratios) / len(ratios)
        if avg_ratio > 1.05:
            headline = f"slim prefill is FASTER than baseline by {avg_ratio:.2f}x on turn 1 (avg across {len(slims)} clients)"
        elif avg_ratio < 0.95:
            headline = f"slim prefill is SLOWER than baseline by {1/avg_ratio:.2f}x on turn 1 (avg across {len(slims)} clients)"
        else:
            headline = f"slim prefill is EQUAL to baseline on turn 1 (ratio={avg_ratio:.2f}, avg across {len(slims)} clients)"
        print(f"\nHeadline: {headline}")

    return 0 if all(summary[c]["verdict"] == "PASS" for c in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
