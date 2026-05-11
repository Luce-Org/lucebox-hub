#!/usr/bin/env python3
"""Parity gate for native MTP integrated decode.

Shells out to `test_dflash --mtp-baseline-check`, parses the three lines it
emits, and either prints a one-line PASS/FAIL summary or writes a JSON
artifact. Used to defend the claim that the integrated MTP decode loop
produces token-identical output to the AR baseline at the same temperature.

This is intentionally a real-model gate (loads the GGUF, runs both decode
modes end-to-end), not a unit test. The C++ side prints these lines that
this script parses:

  [mtp-baseline] baseline   generated=N tok/s=X seconds=Y
  [mtp-baseline] integrated generated=N draft_n=D accepted=A corrected=C \\
                 acceptance=P% tok/s=X seconds=Y draft_n_max=M speed_ratio=Rx
  [mtp-baseline] compare_ok tokens=N mismatches=0
  (or)
  [mtp-baseline] compare_fail mismatch=I baseline=tok integrated=tok ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASELINE_RE = re.compile(
    r"^\[mtp-baseline\]\s+baseline\s+generated=(?P<generated>\d+)\s+"
    r"tok/s=(?P<tps>[0-9.]+)\s+seconds=(?P<seconds>[0-9.]+)",
    re.MULTILINE,
)
INTEGRATED_RE = re.compile(
    r"^\[mtp-baseline\]\s+integrated\s+generated=(?P<generated>\d+)\s+"
    r"draft_n=(?P<draft_n>\d+)\s+accepted=(?P<accepted>\d+)\s+"
    r"corrected=(?P<corrected>\d+)\s+acceptance=(?P<acceptance>[0-9.]+)%\s+"
    r"tok/s=(?P<tps>[0-9.]+)\s+seconds=(?P<seconds>[0-9.]+)\s+"
    r"draft_n_max=(?P<draft_n_max>\d+)\s+speed_ratio=(?P<speed_ratio>[0-9.]+)x",
    re.MULTILINE,
)
COMPARE_OK_RE = re.compile(
    r"^\[mtp-baseline\]\s+compare_ok\s+tokens=(?P<tokens>\d+)\s+mismatches=0",
    re.MULTILINE,
)
COMPARE_FAIL_RE = re.compile(
    r"^\[mtp-baseline\]\s+compare_fail\s+mismatch=(?P<mismatch>\d+)\s+"
    r"baseline=(?P<baseline_tok>-?\d+)\s+integrated=(?P<integrated_tok>-?\d+)",
    re.MULTILINE,
)
TARGET_RE = re.compile(
    r"^\[target\]\s+target loaded:.*trunk_layers=(?P<trunk_layers>\d+)\s+nextn=(?P<nextn>\d+)",
    re.MULTILINE,
)


def _default_binary() -> Path:
    root = Path(__file__).resolve().parents[1]
    if os.name == "nt":
        cand = root / "build" / "test_dflash.exe"
        if cand.exists():
            return cand
        return root / "build" / "Release" / "test_dflash.exe"
    return root / "build" / "test_dflash"


def _write_prompt_ids(out: Path, ids: list[int]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"".join(struct.pack("<i", i) for i in ids))


def _tokenize_via_hf(text: str, tokenizer_id: str) -> list[int]:
    """Tokenize using HuggingFace AutoTokenizer if available."""
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError:
        sys.stderr.write(
            "mtp_baseline_gate: transformers not installed; "
            "pass --prompt-file=<ids.bin> instead of --prompt-text\n"
        )
        sys.exit(2)
    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    return tok.encode(text, add_special_tokens=False)


def _parse_run(stdout: str, stderr: str) -> dict[str, Any]:
    result: dict[str, Any] = {"raw_stdout_lines": stdout.count("\n")}
    m = TARGET_RE.search(stdout)
    if m:
        result["target"] = {
            "trunk_layers": int(m.group("trunk_layers")),
            "nextn": int(m.group("nextn")),
        }
    m = BASELINE_RE.search(stdout)
    if m:
        result["baseline"] = {
            "generated": int(m.group("generated")),
            "tps": float(m.group("tps")),
            "seconds": float(m.group("seconds")),
        }
    m = INTEGRATED_RE.search(stdout)
    if m:
        result["integrated"] = {
            "generated": int(m.group("generated")),
            "draft_n": int(m.group("draft_n")),
            "accepted": int(m.group("accepted")),
            "corrected": int(m.group("corrected")),
            "acceptance_pct": float(m.group("acceptance")),
            "tps": float(m.group("tps")),
            "seconds": float(m.group("seconds")),
            "draft_n_max": int(m.group("draft_n_max")),
            "speed_ratio": float(m.group("speed_ratio")),
        }
    if COMPARE_OK_RE.search(stdout):
        result["compare"] = {"ok": True}
    m = COMPARE_FAIL_RE.search(stderr) or COMPARE_FAIL_RE.search(stdout)
    if m:
        result["compare"] = {
            "ok": False,
            "mismatch": int(m.group("mismatch")),
            "baseline_tok": int(m.group("baseline_tok")),
            "integrated_tok": int(m.group("integrated_tok")),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path,
                        help="Path to the Qwen3.5/3.6-MTP GGUF (am17an-style).")
    parser.add_argument("--prompt-text", default=None,
                        help="Inline prompt text (requires --tokenizer).")
    parser.add_argument("--prompt-file", default=None, type=Path,
                        help="Pre-tokenized prompt file (.bin of i32-le ids).")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B",
                        help="HuggingFace tokenizer id used when --prompt-text is given.")
    parser.add_argument("--n-gen", type=int, default=128)
    parser.add_argument("--mtp-draft-n-max", type=int, default=4)
    parser.add_argument("--max-ctx", type=int, default=4096)
    parser.add_argument("--decode-pos-offset", type=int, default=0)
    parser.add_argument("--fa-window", type=int, default=0,
                        help="DFLASH27B_FA_WINDOW env override; 0 = full attention.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--binary", default=None, type=Path,
                        help="Path to test_dflash binary; defaults to build/test_dflash.")
    parser.add_argument("--min-speed-ratio", type=float, default=1.0,
                        help="Fail if integrated tok/s / baseline tok/s falls below this.")
    parser.add_argument("--require-compare-ok", action="store_true", default=True)
    parser.add_argument("--allow-mismatch", dest="require_compare_ok",
                        action="store_false")
    parser.add_argument("--out", default=None, type=Path,
                        help="Write JSON artifact to this path.")
    parser.add_argument("--step-log", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=1800)
    args = parser.parse_args()

    if not args.target.exists():
        print(f"target GGUF not found: {args.target}", file=sys.stderr)
        return 2
    binary = args.binary or _default_binary()
    if not binary.exists():
        print(f"test_dflash binary not found: {binary}", file=sys.stderr)
        return 2

    # Resolve prompt to a .bin path the binary can consume. Always use an
    # absolute path so the binary's cwd doesn't matter.
    workdir = Path(args.target).resolve().parent if args.target else Path(".").resolve()
    if args.prompt_file is None:
        if not args.prompt_text:
            print("provide --prompt-text or --prompt-file", file=sys.stderr)
            return 2
        ids = _tokenize_via_hf(args.prompt_text, args.tokenizer)
        if not ids:
            print("tokenizer returned empty prompt", file=sys.stderr)
            return 2
        prompt_path = (workdir / "_mtp_baseline_gate_prompt.bin").resolve()
        _write_prompt_ids(prompt_path, ids)
    else:
        prompt_path = Path(args.prompt_file).resolve()
        if not prompt_path.exists():
            print(f"prompt file not found: {prompt_path}", file=sys.stderr)
            return 2

    env = os.environ.copy()
    if args.fa_window > 0:
        env["DFLASH27B_FA_WINDOW"] = str(args.fa_window)
    if "CUDA_VISIBLE_DEVICES" not in env and args.gpu >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cmd = [
        str(binary), str(args.target),
        "--mtp-integrated", "--mtp-baseline-check",
        f"--mtp-draft-n-max={args.mtp_draft_n_max}",
        f"--max-ctx={args.max_ctx}",
        f"--decode-pos-offset={args.decode_pos_offset}",
        f"--prompt-file={prompt_path}",
        f"--n-gen={args.n_gen}",
    ]
    if args.step_log:
        cmd.append("--mtp-step-log")

    print(f"[gate] running: {' '.join(cmd)}", flush=True)
    t0 = datetime.now(timezone.utc)
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=args.timeout_s,
    )
    t1 = datetime.now(timezone.utc)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    parsed = _parse_run(proc.stdout, proc.stderr)
    parsed["cmd"] = cmd
    parsed["exit_code"] = proc.returncode
    parsed["wall_seconds"] = (t1 - t0).total_seconds()
    parsed["target_path"] = str(args.target)
    parsed["n_gen"] = args.n_gen
    parsed["mtp_draft_n_max"] = args.mtp_draft_n_max
    parsed["fa_window"] = args.fa_window
    parsed["decode_pos_offset"] = args.decode_pos_offset
    parsed["min_speed_ratio"] = args.min_speed_ratio

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(parsed, indent=2))
        print(f"[gate] wrote {args.out}")

    # Gate decisions.
    ok_exit = proc.returncode == 0
    compare = parsed.get("compare") or {}
    compare_ok = bool(compare.get("ok"))
    integrated = parsed.get("integrated") or {}
    ratio = float(integrated.get("speed_ratio", 0.0))
    speed_ok = ratio >= args.min_speed_ratio

    if args.require_compare_ok and not compare_ok:
        print(f"[gate] FAIL parity (compare={compare})", file=sys.stderr)
        return 1
    if not ok_exit:
        print(f"[gate] FAIL exit_code={proc.returncode}", file=sys.stderr)
        return proc.returncode if proc.returncode != 0 else 1
    if not speed_ok:
        print(f"[gate] FAIL speed: ratio={ratio:.3f}x < min={args.min_speed_ratio:.3f}x",
              file=sys.stderr)
        return 1

    print(f"[gate] PASS compare_ok=True speed_ratio={ratio:.3f}x "
          f"(min={args.min_speed_ratio:.3f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
