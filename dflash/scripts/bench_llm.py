"""
10 prompts per dataset, AR + DFlash per prompt.

    python3 scripts/bench_llm.py

Paths resolve from the repo root by default. Override with env vars:
    DFLASH_TARGET    path to target Qwen3.6-27B-Q4_K_M.gguf (or 3.5)
    DFLASH_DRAFT     path to draft model.safetensors
    DFLASH_BIN       path to build/test_dflash
    DFLASH_BIN_AR    path to build/test_generate
    DFLASH_TOKENIZER HF tokenizer repo (default Qwen/Qwen3.6-27B; matches run.py)
"""
import json
import os
import re
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN_SUFFIX = ".exe" if os.name == "nt" else ""
TARGET = os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
)
_LOCAL_DRAFT_FILE = ROOT / "models" / "draft" / "model.safetensors"
_LOCAL_DRAFT_ROOT = ROOT / "models" / "draft"
DRAFT = None
TEST_DFLASH = os.environ.get("DFLASH_BIN", str(ROOT / "build" / f"test_dflash{BIN_SUFFIX}"))
TEST_GENERATE = os.environ.get("DFLASH_BIN_AR", str(ROOT / "build" / f"test_generate{BIN_SUFFIX}"))
TOKENIZER = os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.6-27B")
TMPDIR = Path(tempfile.gettempdir()) / "dflash_bench"
TMPDIR.mkdir(parents=True, exist_ok=True)

N_GEN = 256
BUDGET = 22
N_SAMPLE = 10

BENCHES = [
    ("HumanEval", "openai_humaneval", None, "test", lambda x: x["prompt"]),
    ("GSM8K", "gsm8k", "main", "test", lambda x: f"Question: {x['question']}\nAnswer: "),
    ("Math500", "HuggingFaceH4/MATH-500", None, "test", lambda x: f"Problem: {x['problem']}\nSolution: "),
]


def _find_safetensors(root: Path) -> str | None:
    if root.is_file():
        return str(root)
    if not root.is_dir():
        return None
    for st in root.rglob("model.safetensors"):
        return str(st)
    return None


def _resolve_draft() -> str:
    env = os.environ.get("DFLASH_DRAFT")
    if env:
        found = _find_safetensors(Path(env))
        if found:
            return found
        raise FileNotFoundError(f"DFLASH_DRAFT does not point to model.safetensors: {env}")

    for candidate in (_LOCAL_DRAFT_FILE, _LOCAL_DRAFT_ROOT):
        found = _find_safetensors(candidate)
        if found:
            return found

    raise FileNotFoundError(
        "draft model.safetensors not found. Expected one of:\n"
        f"  - {_LOCAL_DRAFT_FILE}\n"
        "Download it as documented in the README, or set DFLASH_DRAFT to an explicit file or directory."
    )


def _require_file(path: str, label: str):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _run_checked(cmd, timeout: int, label: str) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "<no output>").strip()[-2000:]
        raise RuntimeError(f"{label} exited {r.returncode}: {tail}")
    return r


def tokenize(tok, p, path: Path):
    ids = tok.encode(p, add_special_tokens=False)
    with open(path, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))
    return len(ids)


def run_ar(path: Path):
    out_bin = TMPDIR / "ar_out.bin"
    r = _run_checked(
        [TEST_GENERATE, TARGET, str(path), str(N_GEN), str(out_bin)],
        timeout=300,
        label="test_generate",
    )
    m = re.search(r"(\d+\.\d+)\s+tok/s", r.stdout)
    if not m:
        raise RuntimeError(f"test_generate output parse failed: {r.stdout[-1000:]}")
    return float(m.group(1))


def _auto_max_ctx(n_prompt):
    # Auto-fit attention budget: prompt + gen + small verify pad, aligned to
    # FATTN_KQ_STRIDE=256. Oversizing max_ctx makes attention stride over
    # unused KV and can cost >20× prefill time (32K prompt + --kv-q4 +
    # max_ctx=131072 → 1035s vs 38s at max_ctx=32768). See scripts/run.py.
    pad = 64  # covers q_len=16 + ddtree budget up to 22 with margin
    return ((n_prompt + N_GEN + pad + 255) // 256) * 256


def run_df(path: Path, n_prompt):
    max_ctx = _auto_max_ctx(n_prompt)
    out_bin = TMPDIR / "df_out.bin"
    r = _run_checked(
        [
            TEST_DFLASH,
            TARGET,
            DRAFT,
            str(path),
            str(N_GEN),
            str(out_bin),
            "--fast-rollback",
            "--ddtree",
            f"--ddtree-budget={BUDGET}",
            f"--max-ctx={max_ctx}",
        ],
        timeout=300,
        label="test_dflash",
    )
    tps = re.search(r"(\d+(?:\.\d+)?)\s+tok/s", r.stdout)
    al = re.search(r"avg commit/step=(\d+(?:\.\d+)?)", r.stdout)
    if not (tps and al):
        raise RuntimeError(f"test_dflash output parse failed: {r.stdout[-1500:]}")
    return float(tps.group(1)), float(al.group(1))


def main():
    global DRAFT
    DRAFT = _resolve_draft()
    _require_file(TARGET, "target GGUF")
    _require_file(TEST_DFLASH, "test_dflash binary")
    _require_file(TEST_GENERATE, "test_generate binary")

    print(f"[bench] target    = {TARGET}", flush=True)
    print(f"[bench] draft     = {DRAFT}", flush=True)
    print(f"[bench] ar bin    = {TEST_GENERATE}", flush=True)
    print(f"[bench] df bin    = {TEST_DFLASH}", flush=True)
    print(f"[bench] tokenizer = {TOKENIZER}", flush=True)

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

    results = {}
    for name, ds_name, cfg, split, extract in BENCHES:
        print(f"\n[bench] ==== {name} (n={N_SAMPLE}) ====", flush=True)
        ds = load_dataset(ds_name, cfg, split=split)
        ds = ds.shuffle(seed=42).select(range(N_SAMPLE))
        ar_tps, df_tps, df_al = [], [], []
        for i, s in enumerate(ds):
            p = extract(s)
            path = TMPDIR / f"b_{name}_{i:02d}.bin"
            n = tokenize(tok, p, path)
            if n == 0 or n > 3500:
                continue
            try:
                ar = run_ar(path)
                df, al = run_df(path, n)
            except Exception as e:
                print(f"  [{i+1:02d}/{N_SAMPLE}] n_tok={n:4d}  FAILED: {e}", flush=True)
                continue
            if ar > 0:
                ar_tps.append(ar)
            if df > 0:
                df_tps.append(df)
                df_al.append(al)
            print(f"  [{i+1:02d}/{N_SAMPLE}] n_tok={n:4d}  AR={ar:6.2f}  DFlash={df:7.2f}  AL={al:5.2f}", flush=True)
        ar_m = sum(ar_tps) / len(ar_tps) if ar_tps else 0
        df_m = sum(df_tps) / len(df_tps) if df_tps else 0
        al_m = sum(df_al) / len(df_al) if df_al else 0
        results[name] = {"ar": ar_m, "dflash": df_m, "al": al_m,
                         "speedup": df_m / ar_m if ar_m else 0}
        print(f"  {name} mean: AR={ar_m:.2f}  DFlash={df_m:.2f}  AL={al_m:.2f}  {results[name]['speedup']:.2f}x", flush=True)

    print("\n[bench] === SUMMARY ===")
    print(f"{'Task':12s}  {'AR':>8s}  {'DFlash':>8s}  {'AL':>6s}  {'Speedup':>8s}")
    for name, r in results.items():
        print(f"{name:12s}  {r['ar']:8.2f}  {r['dflash']:8.2f}  {r['al']:6.2f}  {r['speedup']:7.2f}x")

    out_json = TMPDIR / "bench_llm_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bench] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
