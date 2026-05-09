"""
Bench DFlash test_dflash over multiple HumanEval-style prompts to get a stable
average acceptance length. Single-prompt measurements are noisy — z-lab's 8.09
AL on humaneval is averaged over 164 samples.

Usage on lucebox:
    python3 bench_he.py                 # run all 10 prompts with --fast-rollback
    python3 bench_he.py --mode batched  # run without --fast-rollback for A/B
"""
import argparse
import os
import re
import struct
import subprocess
import sys
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
TEST_DFLASH = os.environ.get(
    "DFLASH_BIN",
    str(ROOT / "build" / f"test_dflash{BIN_SUFFIX}"),
)
TMPDIR = Path(tempfile.gettempdir()) / "dflash_bench"
TMPDIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    # (name, source_code)
    (
        "has_close_elements",
        "from typing import List\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        '    """Check if in given list of numbers, are any two numbers closer to each other than\n'
        "    given threshold.\n"
        "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
        "    False\n"
        "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
        "    True\n"
        '    """\n'
        "    for",
    ),
    (
        "separate_paren_groups",
        "from typing import List\n\n"
        "def separate_paren_groups(paren_string: str) -> List[str]:\n"
        '    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n'
        "    separate those group into separate strings and return the list of those.\n"
        "    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n"
        "    Ignore any spaces in the input string.\n"
        "    >>> separate_paren_groups('( ) (( )) (( )( ))')\n"
        "    ['()', '(())', '(()())']\n"
        '    """\n'
        "    result = []\n"
        "    current_string = []\n"
        "    current_depth = 0\n"
        "    for",
    ),
    (
        "truncate_number",
        "def truncate_number(number: float) -> float:\n"
        '    """ Given a positive floating point number, it can be decomposed into\n'
        "    and integer part (largest integer smaller than given number) and decimals\n"
        "    (leftover part always smaller than 1).\n"
        "\n"
        "    Return the decimal part of the number.\n"
        "    >>> truncate_number(3.5)\n"
        "    0.5\n"
        '    """\n'
        "    return",
    ),
    (
        "below_zero",
        "from typing import List\n\n"
        "def below_zero(operations: List[int]) -> bool:\n"
        '    """ You\'re given a list of deposit and withdrawal operations on a bank account that starts with\n'
        "    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n"
        "    at that point function should return True. Otherwise it should return False.\n"
        "    >>> below_zero([1, 2, 3])\n"
        "    False\n"
        "    >>> below_zero([1, 2, -4, 5])\n"
        "    True\n"
        '    """\n'
        "    balance = 0\n"
        "    for op in",
    ),
    (
        "mean_absolute_deviation",
        "from typing import List\n\n"
        "def mean_absolute_deviation(numbers: List[float]) -> float:\n"
        '    """ For a given list of input numbers, calculate Mean Absolute Deviation\n'
        "    around the mean of this dataset.\n"
        "    Mean Absolute Deviation is the average absolute difference between each\n"
        "    element and a centerpoint (mean in this case):\n"
        "    MAD = average | x - x_mean |\n"
        "    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n"
        "    1.0\n"
        '    """\n'
        "    mean =",
    ),
    (
        "intersperse",
        "from typing import List\n\n"
        "def intersperse(numbers: List[int], delimeter: int) -> List[int]:\n"
        "    \"\"\" Insert a number 'delimeter' between every two consecutive elements of input list `numbers'\n"
        "    >>> intersperse([], 4)\n"
        "    []\n"
        "    >>> intersperse([1, 2, 3], 4)\n"
        "    [1, 4, 2, 4, 3]\n"
        '    """\n'
        "    result = []\n"
        "    for i, n in",
    ),
    (
        "parse_nested_parens",
        "from typing import List\n\n"
        "def parse_nested_parens(paren_string: str) -> List[int]:\n"
        '    """ Input to this function is a string represented multiple groups for nested parentheses separated by spaces.\n'
        "    For each of the group, output the deepest level of nesting of parentheses.\n"
        "    E.g. (()()) has maximum two levels of nesting while ((())) has three.\n"
        "    >>> parse_nested_parens('(()()) ((())) () ((())()())')\n"
        "    [2, 3, 1, 3]\n"
        '    """\n'
        "    def parse_paren_group(s):\n"
        "        depth = 0\n"
        "        max_depth = 0\n"
        "        for c in",
    ),
    (
        "filter_by_substring",
        "from typing import List\n\n"
        "def filter_by_substring(strings: List[str], substring: str) -> List[str]:\n"
        '    """ Filter an input list of strings only for ones that contain given substring\n'
        "    >>> filter_by_substring([], 'a')\n"
        "    []\n"
        "    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n"
        "    ['abc', 'bacd', 'array']\n"
        '    """\n'
        "    return",
    ),
    (
        "sum_product",
        "from typing import List, Tuple\n\n"
        "def sum_product(numbers: List[int]) -> Tuple[int, int]:\n"
        '    """ For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.\n'
        "    Empty sum should be equal to 0 and empty product should be equal to 1.\n"
        "    >>> sum_product([])\n"
        "    (0, 1)\n"
        "    >>> sum_product([1, 2, 3, 4])\n"
        "    (10, 24)\n"
        '    """\n'
        "    s = 0\n"
        "    p = 1\n"
        "    for n in",
    ),
    (
        "rolling_max",
        "from typing import List\n\n"
        "def rolling_max(numbers: List[int]) -> List[int]:\n"
        '    """ From a given list of integers, generate a list of rolling maximum element found until given moment\n'
        "    in the sequence.\n"
        "    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n"
        "    [1, 2, 3, 3, 3, 4, 4]\n"
        '    """\n'
        "    result = []\n"
        "    running_max = None\n"
        "    for n in numbers:\n"
        "        if running_max is",
    ),
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


def _tokenizer_slug(tokenizer_id: str) -> str:
    """Filesystem-safe slug for tokenizer cache keying."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tokenizer_id)


def _prompt_path(i: int, tokenizer_slug: str) -> Path:
    return TMPDIR / f"he_prompt_{tokenizer_slug}_{i:02d}.bin"


def tokenize_prompt(prompt: str, out_path: Path, tokenizer) -> int:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    with open(out_path, "wb") as f:
        for tid in ids:
            f.write(struct.pack("<i", int(tid)))
    return len(ids)


def run_test_dflash(prompt_path: Path, n_gen: int, fast_rollback: bool,
                    ddtree_budget: int | None = None,
                    ddtree_temp: float | None = None,
                    ddtree_no_chain_seed: bool = False,
                    extra_args: list[str] | None = None,
                    extra_env: dict[str, str] | None = None) -> dict:
    out_bin = TMPDIR / "he_bench_out.bin"
    cmd = [
        TEST_DFLASH, TARGET, DRAFT, str(prompt_path), str(n_gen), str(out_bin),
    ]
    if fast_rollback:
        cmd.append("--fast-rollback")
    if ddtree_budget is not None:
        cmd.append("--ddtree")
        cmd.append(f"--ddtree-budget={ddtree_budget}")
    if ddtree_temp is not None:
        cmd.append(f"--ddtree-temp={ddtree_temp}")
    if ddtree_no_chain_seed:
        cmd.append("--ddtree-no-chain-seed")
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
        raise RuntimeError(f"test_dflash exited {r.returncode}")

    # Parse output
    out = r.stdout
    m_tps = re.search(r"(\d+(?:\.\d+)?)\s+tok/s", out)
    m_commit = re.search(r"avg commit/step=(\d+(?:\.\d+)?)", out)
    m_accept = re.search(r"accepted=(\d+)/(\d+) \((\d+(?:\.\d+)?)%", out)
    m_steps = re.search(r"(\d+) draft steps", out)
    if not (m_tps and m_commit and m_accept and m_steps):
        print("STDOUT tail:", out[-2000:])
        raise RuntimeError("failed to parse output")
    return {
        "tok_s": float(m_tps.group(1)),
        "commit_per_step": float(m_commit.group(1)),
        "accepted": int(m_accept.group(1)),
        "total_draft_pos": int(m_accept.group(2)),
        "pct": float(m_accept.group(3)),
        "steps": int(m_steps.group(1)),
    }


def main():
    global DRAFT
    DRAFT = _resolve_draft()
    _require_file(TARGET, "target GGUF")
    _require_file(TEST_DFLASH, "test_dflash binary")

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--mode", choices=["fast", "batched"], default="fast")
    ap.add_argument("--skip-tokenize", action="store_true")
    ap.add_argument("--ddtree-budget", type=int, default=None,
                    help="Enable DDTree mode with this node budget (e.g. 15, 32, 64)")
    ap.add_argument("--ddtree-temp", type=float, default=None,
                    help="Sharpen draft logits with this temperature (T<1 widens top-1/top-2 gap)")
    ap.add_argument("--ddtree-no-chain-seed", action="store_true",
                    help="Use paper's pure best-first (no chain pre-seed)")
    ap.add_argument("--draft-feature-mirror", action="store_true",
                    help="Use the draft-side target feature mirror path")
    ap.add_argument("--target-gpu", type=int, default=None,
                    help="Visible CUDA device id for the target backend")
    ap.add_argument("--draft-gpu", type=int, default=None,
                    help="Visible CUDA device id for the draft backend")
    ap.add_argument("--cuda-visible-devices", default=None,
                    help="Optional CUDA_VISIBLE_DEVICES override for test_dflash")
    ap.add_argument("--target-tokenizer",
                    default=os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.6-27B"),
                    help="HuggingFace tokenizer repo for the target. Defaults to "
                         "$DFLASH_TOKENIZER, then Qwen/Qwen3.6-27B.")
    args = ap.parse_args()

    # Tokenized prompts are cached at TMPDIR/he_prompt_<slug>_NN.bin so
    # different --target-tokenizer values never collide. Without the slug,
    # `--skip-tokenize` after a prior run with a different tokenizer would
    # silently feed the wrong token IDs to the bench.
    tok_slug = _tokenizer_slug(args.target_tokenizer)

    print(f"[bench] target    = {TARGET}")
    print(f"[bench] draft     = {DRAFT}")
    print(f"[bench] bin       = {TEST_DFLASH}")
    print(f"[bench] tmp       = {TMPDIR}")
    print(f"[bench] tokenizer = {args.target_tokenizer}")

    if not args.skip_tokenize:
        print(f"[bench] tokenizing prompts via HF…")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.target_tokenizer, trust_remote_code=True)
        for i, (name, p) in enumerate(PROMPTS):
            path = _prompt_path(i, tok_slug)
            n = tokenize_prompt(p, path, tok)
            print(f"  [{i:02d}] {name:26s}  {n:4d} tokens")
    else:
        if not _prompt_path(0, tok_slug).exists():
            sys.exit(
                f"[error] --skip-tokenize requested but no cache for "
                f"tokenizer={args.target_tokenizer!r} (looked for "
                f"{_prompt_path(0, tok_slug)}). Drop --skip-tokenize to "
                f"tokenize fresh, or pass --target-tokenizer matching a "
                f"previous run.")
        print(f"[bench] skipping tokenize (reusing {_prompt_path(0, tok_slug).parent})")

    print(f"\n[bench] mode={args.mode}  n_gen={args.n_gen}")
    print(f"{'prompt':28s}  {'steps':>6s} {'AL':>6s} {'pct%':>6s} {'tok/s':>8s}")
    print("-" * 62)

    extra_args = []
    if args.draft_feature_mirror:
        extra_args.append("--draft-feature-mirror")
    if args.target_gpu is not None:
        extra_args.append(f"--target-gpu={args.target_gpu}")
    if args.draft_gpu is not None:
        extra_args.append(f"--draft-gpu={args.draft_gpu}")

    extra_env = {}
    if args.cuda_visible_devices:
        extra_env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    results = []
    for i, (name, _) in enumerate(PROMPTS):
        path = _prompt_path(i, tok_slug)
        try:
            r = run_test_dflash(path, args.n_gen,
                                fast_rollback=(args.mode == "fast"),
                                ddtree_budget=args.ddtree_budget,
                                ddtree_temp=args.ddtree_temp,
                                ddtree_no_chain_seed=args.ddtree_no_chain_seed,
                                extra_args=extra_args,
                                extra_env=extra_env)
        except Exception as e:
            print(f"  [{i:02d}] {name:26s}  FAILED: {e}")
            continue
        results.append((name, r))
        print(
            f"  {name:26s}  {r['steps']:6d} {r['commit_per_step']:6.2f} "
            f"{r['pct']:6.1f} {r['tok_s']:8.2f}"
        )

    if not results:
        print("no successful runs")
        sys.exit(1)

    n = len(results)
    mean_al = sum(r["commit_per_step"] for _, r in results) / n
    mean_tps = sum(r["tok_s"] for _, r in results) / n
    mean_pct = sum(r["pct"] for _, r in results) / n

    print("-" * 62)
    print(f"{'MEAN':28s}  {'':6s} {mean_al:6.2f} {mean_pct:6.1f} {mean_tps:8.2f}")
    print()
    print(f"commit/step range: {min(r['commit_per_step'] for _,r in results):.2f} - "
          f"{max(r['commit_per_step'] for _,r in results):.2f}")
    print(f"tok/s range:        {min(r['tok_s'] for _,r in results):.1f} - "
          f"{max(r['tok_s'] for _,r in results):.1f}")


if __name__ == "__main__":
    main()
