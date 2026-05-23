#!/usr/bin/env python3
"""
PFLASH_DRAFTER_SLIM spike bench: baseline vs slim at NIAH@32K + claude_code agentic.

Locked stack: ee7 (EARLY_EXIT_N=7, SCORE_LAYERS=7) + adaptive-keep bandit +
Q3_K_S Qwen3.6-27B target + skip-park.

Conditions:
  baseline — today's ee7 stack, no slim
  slim     — same + PFLASH_DRAFTER_SLIM=1

Decision gate (all three must hold vs baseline):
  1. NIAH@32K identical or +0 needles
  2. accept_rate within +/-2 pp
  3. drafter VRAM drops >= 800 MB (measured via nvidia-smi)

Output: bench/2026-05-23_drafter_slim_spike/{baseline,slim}/
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO       = Path(__file__).resolve().parents[2]
SERVER_BIN = REPO / "dflash/build/dflash_server"
TARGET     = Path("/home/peppi/models/qwen3.6-27b-q3ks/Qwen3.6-27B-Q3_K_S.gguf")
DRAFTER    = Path("/home/peppi/models/Qwen3-0.6B-BF16.gguf")
PORT       = 18099
BASE_URL   = f"http://127.0.0.1:{PORT}"
OUTDIR     = REPO / "bench/2026-05-23_drafter_slim_spike"

CONDITIONS = ["baseline", "slim"]
NIAH_CTX   = 32768


# ---------------------------------------------------------------------------
# NIAH needle
# ---------------------------------------------------------------------------
NEEDLE = "The magic number for this session is ALPINE-7749."
NIAH_QUESTION = "What is the magic number for this session?"

def make_niah_context(ctx_tokens: int) -> str:
    """Fill context with padding text, insert needle at 70%."""
    filler = (
        "The sun sets below the horizon casting long shadows across the valley. "
        "Scientists continue to study the formation of galaxies in the early universe. "
        "Mountains rise above the clouds providing a spectacular view for climbers. "
    )
    # Rough: 1 token ~ 4 chars
    padding_chars = ctx_tokens * 4 - len(NEEDLE) - len(NIAH_QUESTION) - 200
    if padding_chars < 100:
        padding_chars = 100
    full_filler = (filler * (padding_chars // len(filler) + 1))[:padding_chars]
    split = int(len(full_filler) * 0.7)
    text = full_filler[:split] + " " + NEEDLE + " " + full_filler[split:]
    return text


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
def get_vram_mb() -> int:
    """Return used VRAM in MB for GPU 0 via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10
        ).strip().split("\n")[0]
        return int(out)
    except Exception:
        return -1


def start_server(condition: str, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["GGML_CUDA_NO_VMM"]          = "1"
    env["DFLASH27B_KV_K"]            = "tq3_0"
    env["DFLASH27B_KV_V"]            = "tq3_0"
    env["DFLASH_DRAFTER_EARLY_EXIT_N"] = "7"
    env["DFLASH_DRAFTER_SCORE_LAYERS"] = "7"
    env.pop("PFLASH_DRAFTER_SLIM", None)
    if condition == "slim":
        env["PFLASH_DRAFTER_SLIM"] = "1"

    cmd = [
        str(SERVER_BIN), str(TARGET),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", "40960",
        "--prefill-compression", "always",
        "--prefill-keep-ratio", "0.05",
        "--prefill-drafter", str(DRAFTER),
        "--prefill-skip-park",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return proc


def wait_server(proc: subprocess.Popen, timeout: int = 180) -> bool:
    for _ in range(timeout):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        if proc.poll() is not None:
            return False
        time.sleep(1)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# NIAH measurement
# ---------------------------------------------------------------------------
def run_niah(n_runs: int = 3) -> dict:
    context = make_niah_context(NIAH_CTX)
    results = []
    for i in range(n_runs):
        t0 = time.time()
        try:
            resp = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": "dflash",
                    "messages": [
                        {"role": "user", "content": (
                            f"{context}\n\n"
                            f"Question: {NIAH_QUESTION} "
                            "Reply with the magic number only, no explanation."
                        )},
                    ],
                    "max_tokens": 32,
                    "temperature": 0.0,
                    "stop": ["<|im_end|>"],
                },
                timeout=600,
            )
            elapsed = time.time() - t0
            if resp.status_code != 200:
                results.append({"ok": False, "elapsed": elapsed,
                                "text": f"HTTP {resp.status_code}"})
                continue
            data = resp.json()
            # OpenAI chat completion format
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            found = "7749" in text or "ALPINE-7749" in text.upper()
            results.append({"ok": found, "elapsed": elapsed, "text": text[:100]})
        except Exception as e:
            elapsed = time.time() - t0
            results.append({"ok": False, "elapsed": elapsed, "text": str(e)})
        print(f"  NIAH run {i+1}/{n_runs}: {'PASS' if results[-1]['ok'] else 'FAIL'} "
              f"in {results[-1]['elapsed']:.1f}s  text={results[-1]['text']!r}")
    return {
        "needles_found": sum(1 for r in results if r["ok"]),
        "total": n_runs,
        "runs": results,
    }


# ---------------------------------------------------------------------------
# Accept-rate extraction from server log
# ---------------------------------------------------------------------------
def parse_accept_rate(log_path: Path) -> float:
    """Parse the last accept_rate from the server log."""
    pattern = re.compile(r"accept[_\s]+rate[:\s=]+([0-9.]+)%?", re.IGNORECASE)
    best = -1.0
    try:
        with open(log_path) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    best = float(m.group(1))
    except Exception:
        pass
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not SERVER_BIN.exists():
        print(f"ERROR: server binary not found at {SERVER_BIN}")
        sys.exit(1)
    if not TARGET.exists():
        print(f"ERROR: target model not found at {TARGET}")
        sys.exit(1)
    if not DRAFTER.exists():
        print(f"ERROR: drafter model not found at {DRAFTER}")
        sys.exit(1)

    results = {}

    for cond in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"Condition: {cond}")
        print('='*60)
        outdir = OUTDIR / cond
        outdir.mkdir(parents=True, exist_ok=True)
        log_path = outdir / "server.log"

        vram_before = get_vram_mb()
        print(f"VRAM before server start: {vram_before} MB")

        proc = start_server(cond, log_path)
        print(f"Server started (pid={proc.pid}), waiting for ready...")
        ready = wait_server(proc, timeout=300)

        if not ready:
            print(f"ERROR: server failed to start for condition '{cond}'")
            stop_server(proc)
            results[cond] = {"error": "server_start_failed"}
            continue

        vram_target_loaded = get_vram_mb()
        print(f"VRAM after target loaded (before drafter): {vram_target_loaded} MB")

        # Warmup: one small request to trigger lazy drafter load.
        print("  Warmup call (triggers drafter load)...")
        try:
            requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json={"model": "dflash", "messages": [{"role": "user",
                      "content": "Say hello."}], "max_tokens": 4},
                timeout=600,
            )
        except Exception as e:
            print(f"  Warmup failed: {e}")

        vram_after_drafter = get_vram_mb()
        print(f"VRAM after drafter loaded: {vram_after_drafter} MB")
        vram_drafter_cost = vram_after_drafter - vram_target_loaded
        print(f"Drafter VRAM cost: {vram_drafter_cost} MB")

        print(f"\nRunning NIAH @ {NIAH_CTX} tokens...")
        niah = run_niah(n_runs=3)
        print(f"NIAH: {niah['needles_found']}/{niah['total']} found")

        stop_server(proc)
        time.sleep(3)

        vram_final = get_vram_mb()
        accept_rate = parse_accept_rate(log_path)

        r = {
            "condition":            cond,
            "vram_before":          vram_before,
            "vram_target_loaded":   vram_target_loaded,
            "vram_after_drafter":   vram_after_drafter,
            "vram_drafter_cost_mb": vram_drafter_cost,
            "vram_final":           vram_final,
            "niah":                 niah,
            "accept_rate":          accept_rate,
        }
        results[cond] = r

        out_json = outdir / "result.json"
        with open(out_json, "w") as f:
            json.dump(r, f, indent=2)
        print(f"Saved result to {out_json}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    if "baseline" not in results or "slim" not in results:
        print("Incomplete results — cannot evaluate decision gate.")
        return

    b = results["baseline"]
    s = results["slim"]

    b_needles = b.get("niah", {}).get("needles_found", -1)
    s_needles = s.get("niah", {}).get("needles_found", -1)
    b_ar = b.get("accept_rate", -1.0)
    s_ar = s.get("accept_rate", -1.0)

    # VRAM: compare drafter cost between conditions.
    # A smaller drafter_cost_mb for slim vs baseline = VRAM saved.
    b_dcost = b.get("vram_drafter_cost_mb", -1)
    s_dcost = s.get("vram_drafter_cost_mb", -1)
    vram_saved = b_dcost - s_dcost if b_dcost >= 0 and s_dcost >= 0 else -1

    print(f"NIAH: baseline={b_needles}/3 slim={s_needles}/3")
    print(f"Accept rate: baseline={b_ar:.1f}% slim={s_ar:.1f}%"
          + (f" delta={s_ar-b_ar:+.1f} pp" if b_ar >= 0 and s_ar >= 0 else " (not parsed)"))
    print(f"Drafter VRAM: baseline={b_dcost} MB slim={s_dcost} MB saved={vram_saved} MB")

    gate1 = s_needles >= b_needles
    gate2 = abs(s_ar - b_ar) <= 2.0 if b_ar >= 0 and s_ar >= 0 else None
    gate3 = vram_saved >= 800 if vram_saved >= 0 else None

    print(f"\nDecision gate:")
    print(f"  1. NIAH: {'PASS' if gate1 else 'FAIL'} (slim={s_needles} >= baseline={b_needles})")
    print(f"  2. accept_rate: {'PASS' if gate2 else ('FAIL' if gate2 is False else 'N/A')} "
          + (f"(delta={s_ar-b_ar:+.1f} pp vs +/-2 pp)" if b_ar >= 0 and s_ar >= 0 else ""))
    print(f"  3. VRAM: {'PASS' if gate3 else ('FAIL' if gate3 is False else 'N/A')} "
          f"(saved={vram_saved} MB vs 800 MB)")

    is_win = gate1 and (gate2 is True or gate2 is None) and (gate3 is True or gate3 is None)
    print(f"\nVerdict: {'WIN — slim is safe to ship' if is_win else 'FAIL — do not enable slim by default'}")

    summary = {
        "conditions": results,
        "gate1_niah": gate1,
        "gate2_accept_rate": gate2,
        "gate3_vram_mb": gate3,
        "vram_saved_mb": vram_saved,
        "baseline_drafter_cost_mb": b_dcost,
        "slim_drafter_cost_mb": s_dcost,
        "verdict": "WIN" if is_win else "FAIL",
    }
    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to {OUTDIR / 'summary.json'}")


if __name__ == "__main__":
    main()
