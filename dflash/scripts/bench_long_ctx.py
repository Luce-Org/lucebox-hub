"""
Long-context sweep: compare Q4_0 vs TQ3_0 KV cache at 32K / 64K / 128K.

Measures prefill time and DFlash decode tok/s at each context length under
both KV formats. The point is not throughput (TQ3 is slightly slower than
Q4_0 at short contexts) but memory — TQ3_0 (3.5 bpv) uses 22% less KV than
Q4_0 (4.5 bpv), enabling longer contexts on the same VRAM budget.

Usage:
    uv run scripts/bench_long_ctx.py
    uv run scripts/bench_long_ctx.py --ctx 32768 65536
    uv run scripts/bench_long_ctx.py --ctx 131072 --kv tq3

Layer-segmented prefill (DFLASH27B_LAYER_PREFILL=1) is used automatically for
prompts over 8K tokens to reduce peak activation memory.
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN_SUFFIX = ".exe" if os.name == "nt" else ""
TARGET = os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.5-27B-Q4_K_M.gguf"),
)
_LOCAL_DRAFT_ROOT = ROOT / "models" / "draft"
DRAFT = None
TEST_DFLASH = os.environ.get(
    "DFLASH_BIN",
    str(ROOT / "build" / f"test_dflash{BIN_SUFFIX}"),
)
TMPDIR = Path("/tmp/dflash_bench")
TMPDIR.mkdir(parents=True, exist_ok=True)

N_GEN = 128       # shorter gen for long-ctx; prefill dominates
BUDGET = 16       # lower budget reduces per-token SSM memory at long ctx


def _resolve_draft() -> str:
    for candidate in (_LOCAL_DRAFT_ROOT / "model.safetensors", _LOCAL_DRAFT_ROOT):
        if candidate.is_file():
            return str(candidate)
        if candidate.is_dir():
            for st in candidate.rglob("model.safetensors"):
                return str(st)
    raise FileNotFoundError("draft model.safetensors not found under models/draft/")


def _require_file(path: str, label: str):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def make_long_prompt(tok, target_len: int) -> list:
    """Return a list of token IDs of approximately target_len tokens."""
    # Use a realistic long-form passage so KV patterns are diverse.
    # Repeat until we have enough, then trim.
    chunk = (
        "Large language models use attention mechanisms to relate tokens across "
        "long contexts. The key-value cache stores intermediate representations "
        "for each token, allowing efficient autoregressive generation. "
        "Quantizing this cache trades a small quality loss for substantial "
        "memory savings, enabling longer effective context windows on the same "
        "GPU. Speculative decoding with a draft model accelerates generation "
        "by proposing multiple tokens per step and verifying them in parallel "
        "against the target model. The acceptance length—average tokens committed "
        "per verify step—determines the effective speedup. "
    )
    chunk_ids = tok.encode(chunk, add_special_tokens=False)
    repeats = (target_len // len(chunk_ids)) + 2
    ids = (chunk_ids * repeats)[:target_len]
    return ids


def write_prompt_bin(ids: list, path: Path) -> int:
    with open(path, "wb") as f:
        for tok_id in ids:
            f.write(struct.pack("<i", int(tok_id)))
    return len(ids)


def run_dflash(prompt_path: Path, n_prompt: int, kv_mode: str) -> dict:
    """Run test_dflash and return parsed stats."""
    pad = 64
    max_ctx = ((n_prompt + N_GEN + pad + 255) // 256) * 256

    out_bin = TMPDIR / f"lc_{kv_mode}_out.bin"
    env = dict(os.environ)
    env.pop("DFLASH27B_KV_Q4", None)
    env.pop("DFLASH27B_KV_TQ3", None)
    if kv_mode == "q4":
        env["DFLASH27B_KV_Q4"] = "1"
    elif kv_mode == "tq3":
        env["DFLASH27B_KV_TQ3"] = "1"
    # Use layer-segmented prefill for long contexts — reduces peak activation
    # memory and avoids graph rebuild overhead at full context.
    if n_prompt > 8192:
        env["DFLASH27B_LAYER_PREFILL"] = "1"
    else:
        env.pop("DFLASH27B_LAYER_PREFILL", None)

    cmd = [
        TEST_DFLASH, TARGET, DRAFT,
        str(prompt_path), str(N_GEN), str(out_bin),
        "--fast-rollback", "--ddtree",
        f"--ddtree-budget={BUDGET}",
        f"--max-ctx={max_ctx}",
    ]
    print(f"    cmd: ... --ddtree-budget={BUDGET} --max-ctx={max_ctx}", flush=True)

    t0 = time.monotonic()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    elapsed = time.monotonic() - t0

    if r.returncode != 0:
        print(f"STDERR tail:\n{r.stderr[-3000:]}")
        raise RuntimeError(f"test_dflash exited {r.returncode}")

    out = r.stdout
    # Parse prefill: "[prefill] N tokens in T s" (token-seg or layer-seg)
    m_pf = re.search(r"\[prefill\](?:\s+layer-seg)?\s+(\d+)\s+tokens in\s+([\d.]+)\s+s", out)
    m_tps = re.search(r"([\d.]+)\s+tok/s", out)
    m_al = re.search(r"avg commit/step=([\d.]+)", out)

    if not (m_pf and m_tps and m_al):
        print("STDOUT tail:", out[-4000:])
        raise RuntimeError("failed to parse test_dflash output")

    # Estimate KV size based on Qwen3.5-27B architecture:
    # 28 layers, head_dim=128, n_kv_heads=4, 2 tensors (K+V) per layer.
    # F16 reference: max_ctx * 28 * 4 * 2 * 128 * 2 bytes
    f16_bytes = max_ctx * 28 * 4 * 2 * 128 * 2
    bpv = {"f16": 16, "q4": 4.5, "tq3": 3.5}.get(kv_mode, 16)
    kv_gb = f16_bytes * bpv / 16 / 1e9

    return {
        "n_prompt": n_prompt,
        "max_ctx": max_ctx,
        "kv_mode": kv_mode,
        "prefill_s": float(m_pf.group(2)),
        "tok_s": float(m_tps.group(1)),
        "al": float(m_al.group(1)),
        "kv_gb": round(kv_gb, 2),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    global DRAFT
    DRAFT = _resolve_draft()
    _require_file(TARGET, "target GGUF")
    _require_file(TEST_DFLASH, "test_dflash binary")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ctx", nargs="+", type=int,
        default=[32768, 65536, 131072],
        help="Prompt lengths to test (tokens). Default: 32K 64K 128K",
    )
    ap.add_argument(
        "--kv", nargs="+", choices=["q4", "tq3", "f16"],
        default=["q4", "tq3"],
        help="KV cache formats to compare. Default: q4 tq3",
    )
    ap.add_argument(
        "--skip-tokenize", action="store_true",
        help="Reuse cached prompt .bin files from a previous run",
    )
    args = ap.parse_args()

    print(f"[bench] target = {TARGET}")
    print(f"[bench] draft  = {DRAFT}")
    print(f"[bench] bin    = {TEST_DFLASH}")
    print(f"[bench] ctx_lengths = {args.ctx}")
    print(f"[bench] kv_modes    = {args.kv}")
    print(f"[bench] n_gen={N_GEN}  ddtree_budget={BUDGET}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", trust_remote_code=True)

    results = []
    for ctx_len in args.ctx:
        prompt_path = TMPDIR / f"long_ctx_{ctx_len}.bin"
        if args.skip_tokenize and prompt_path.exists():
            n = ctx_len
            print(f"\n[bench] ctx={ctx_len}: reusing {prompt_path}")
        else:
            ids = make_long_prompt(tok, ctx_len)
            n = write_prompt_bin(ids, prompt_path)
            print(f"\n[bench] ctx={ctx_len}: wrote {n} tokens → {prompt_path}")

        for kv in args.kv:
            print(f"\n[bench] ctx={ctx_len:>7d}  kv={kv} ...", flush=True)
            try:
                r = run_dflash(prompt_path, n, kv)
                results.append(r)
                print(
                    f"  prefill={r['prefill_s']:6.1f}s  decode={r['tok_s']:6.1f} tok/s  "
                    f"AL={r['al']:.2f}  KV≈{r['kv_gb']:.2f} GB"
                )
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({
                    "n_prompt": ctx_len, "kv_mode": kv,
                    "error": str(e),
                })

    out_json = TMPDIR / "long_ctx_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[bench] wrote {out_json}")

    # Summary table
    print("\n=== Long-context sweep summary (RTX 5090 Laptop) ===")
    print()
    print(f"{'Ctx':>8s}  {'KV':>5s}  {'Prefill':>8s}  {'Decode tok/s':>13s}  "
          f"{'AL':>5s}  {'KV size':>8s}")
    print("-" * 58)
    for r in results:
        if "error" in r:
            print(f"{r['n_prompt']//1024:>7d}K  {r['kv_mode']:>5s}  {'FAILED':>8s}  "
                  f"{r['error'][:30]}")
            continue
        print(
            f"{r['n_prompt']//1024:>7d}K  {r['kv_mode']:>5s}  "
            f"{r['prefill_s']:>7.1f}s  {r['tok_s']:>12.1f}  "
            f"{r['al']:>5.2f}  {r['kv_gb']:>6.2f} GB"
        )


if __name__ == "__main__":
    main()
