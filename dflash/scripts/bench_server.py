"""
OpenAI-compat server throughput benchmark — code-agent prompt mix.

Streams each prompt through /v1/chat/completions and records:
  TTFT   — time to first content token (ms)
  tok/s  — decode throughput (completion_tokens / decode_wall_time)
  n_tok  — completion token count (from usage chunk)

Usage:
    # dflash (default)
    uv run scripts/bench_server.py

    # LM Studio or any other endpoint
    uv run scripts/bench_server.py --url http://localhost:1234 --model <model-id>

    # Repeat each prompt N times for stable averages
    uv run scripts/bench_server.py --repeat 3 --n-gen 256
"""
import argparse
import json
import time
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass, field


# ── Prompt corpus ─────────────────────────────────────────────────
# Typical single-turn code-agent requests: short system + focused user ask.
# Kept concise so prompt tokens stay small and decode throughput dominates.

SYSTEM = (
    "You are a senior software engineer. "
    "Reply with working code and brief inline comments. "
    "No preamble, no markdown fences unless the output is a complete file."
)

PROMPTS = [
    (
        "refactor_function",
        "Refactor this Python function so it handles empty input gracefully "
        "and uses a list comprehension instead of the explicit loop:\n\n"
        "def doubled(nums):\n    result = []\n    for n in nums:\n        result.append(n * 2)\n    return result",
    ),
    (
        "write_tests",
        "Write pytest unit tests for this function:\n\n"
        "def clamp(value, lo, hi):\n    return max(lo, min(hi, value))\n\n"
        "Cover: normal range, below lo, above hi, lo==hi, negative values.",
    ),
    (
        "fix_bug",
        "Fix the off-by-one error in this binary search and explain the change:\n\n"
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)\n"
        "    while lo < hi:\n        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target: return mid\n"
        "        elif arr[mid] < target: lo = mid\n"
        "        else: hi = mid\n    return -1",
    ),
    (
        "explain_code",
        "Explain what this decorator does and when you'd use it:\n\n"
        "import functools\n"
        "def retry(n=3):\n    def decorator(fn):\n        @functools.wraps(fn)\n"
        "        def wrapper(*args, **kwargs):\n            for i in range(n):\n"
        "                try: return fn(*args, **kwargs)\n"
        "                except Exception:\n                    if i == n-1: raise\n"
        "        return wrapper\n    return decorator",
    ),
    (
        "add_feature",
        "Add an optional `timeout` parameter (default 30s) to this HTTP GET helper. "
        "Raise a clear TimeoutError with the URL in the message:\n\n"
        "import urllib.request\ndef fetch(url: str) -> bytes:\n"
        "    with urllib.request.urlopen(url) as r:\n        return r.read()",
    ),
    (
        "code_review",
        "Review this Go snippet for correctness and idiomatic style. "
        "Give concrete suggestions, not general advice:\n\n"
        "func sum(nums []int) int {\n    total := 0\n    for i := 0; i < len(nums); i++ {\n"
        "        total = total + nums[i]\n    }\n    return total\n}",
    ),
    (
        "shell_command",
        "Give me a single bash one-liner to find all Python files modified in the last "
        "7 days under ./src, print their paths, and count them at the end.",
    ),
    (
        "generate_schema",
        "Write a Pydantic v2 model for a REST API response that contains: "
        "a string `id`, an ISO-8601 `created_at` datetime, a list of `tags` (strings), "
        "and an optional `metadata` dict. Add field validators where useful.",
    ),
]


# ── Streaming SSE reader ───────────────────────────────────────────

@dataclass
class ProbResult:
    name: str
    ttft_ms: float
    decode_tok_s: float
    n_tok: int
    error: str = ""


def _stream_prompt(url: str, model: str, system: str, user: str,
                   n_gen: int) -> ProbResult:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": n_gen,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()

    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    t_first: float | None = None
    n_tok = 0
    buf = b""

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            while True:
                chunk = resp.read(256)
                if not chunk:
                    break
                buf += chunk
                while b"\n\n" in buf:
                    line, buf = buf.split(b"\n\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload_str = line[5:].strip()
                    if payload_str == b"[DONE]":
                        break
                    try:
                        ev = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue

                    # Usage chunk (choices=[])
                    if not ev.get("choices") and ev.get("usage"):
                        n_tok = ev["usage"].get("completion_tokens", n_tok)
                        continue

                    choices = ev.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    # First real content token → record TTFT
                    if t_first is None and (
                        delta.get("content") or delta.get("tool_calls")
                    ):
                        t_first = time.perf_counter()

    except urllib.error.URLError as e:
        return ProbResult(name="", ttft_ms=0, decode_tok_s=0, n_tok=0, error=str(e))

    t_end = time.perf_counter()

    if t_first is None:
        t_first = t_end  # no content token received

    ttft_ms = (t_first - t0) * 1000
    decode_wall = t_end - t_first
    decode_tok_s = n_tok / decode_wall if decode_wall > 0 else 0.0

    return ProbResult(name="", ttft_ms=ttft_ms, decode_tok_s=decode_tok_s, n_tok=n_tok)


# ── Main ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Server throughput benchmark (code-agent prompts)")
    ap.add_argument("--url",    default="http://localhost:1236",
                    help="Base URL of the OpenAI-compat server (default: %(default)s)")
    ap.add_argument("--model",  default="",
                    help="Model id to send in the request (default: auto-detect from /v1/models)")
    ap.add_argument("--n-gen",  type=int, default=512,
                    help="max_tokens per request (default: %(default)s)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Repeat each prompt N times and average (default: %(default)s)")
    ap.add_argument("--prompts", nargs="+", metavar="NAME",
                    help="Run only named prompts (default: all)")
    args = ap.parse_args()

    # Auto-detect model id if not given
    model = args.model
    if not model:
        try:
            with urllib.request.urlopen(f"{args.url}/v1/models", timeout=10) as r:
                data = json.loads(r.read())
            model = data["data"][0]["id"]
            print(f"model: {model} (auto-detected)", file=sys.stderr)
        except Exception as e:
            sys.exit(f"Cannot reach {args.url}/v1/models: {e}\nPass --model explicitly.")

    corpus = [(n, p) for n, p in PROMPTS
              if not args.prompts or n in args.prompts]
    if not corpus:
        sys.exit(f"No prompts matched {args.prompts}")

    print(f"\nBenchmarking {args.url}  model={model}  n_gen={args.n_gen}  repeat={args.repeat}")
    print(f"{'prompt':<25}  {'TTFT ms':>8}  {'tok/s':>7}  {'n_tok':>6}")
    print("-" * 56)

    all_tok_s: list[float] = []
    all_ttft:  list[float] = []

    for name, user in corpus:
        run_tok_s: list[float] = []
        run_ttft:  list[float] = []
        run_ntok:  list[int]   = []
        err = ""
        for _ in range(args.repeat):
            r = _stream_prompt(args.url, model, SYSTEM, user, args.n_gen)
            if r.error:
                err = r.error
                break
            run_tok_s.append(r.decode_tok_s)
            run_ttft.append(r.ttft_ms)
            run_ntok.append(r.n_tok)

        if err:
            print(f"{name:<25}  ERROR: {err}")
            continue

        avg_tok_s = sum(run_tok_s) / len(run_tok_s)
        avg_ttft  = sum(run_ttft)  / len(run_ttft)
        avg_ntok  = sum(run_ntok)  // len(run_ntok)
        all_tok_s.append(avg_tok_s)
        all_ttft.append(avg_ttft)
        print(f"{name:<25}  {avg_ttft:>8.0f}  {avg_tok_s:>7.1f}  {avg_ntok:>6}")

    if all_tok_s:
        print("-" * 56)
        print(f"{'mean':<25}  {sum(all_ttft)/len(all_ttft):>8.0f}  "
              f"{sum(all_tok_s)/len(all_tok_s):>7.1f}")
        print(f"{'min/max tok/s':<25}  {'':>8}  "
              f"{min(all_tok_s):>5.1f} / {max(all_tok_s):.1f}")


if __name__ == "__main__":
    main()
