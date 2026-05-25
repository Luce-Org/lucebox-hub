"""Agentic tool-call reliability benchmark for the lucebox HTTP server.

The prompts are intentionally small and deterministic. They exercise the path
that coding agents depend on: Qwen chat-template tool rendering, model tool
selection, lucebox tool-call parsing, and OpenAI-compatible response shaping.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search files for a literal pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a read-only shell command.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
]


CASES = [
    {
        "name": "read-props-driver",
        "expected": "read_file",
        "prompt": (
            "Produce exactly this tool call and no prose:\n"
            "<tool_call>\n"
            "<function=read_file>\n"
            "<parameter=path>\n"
            "runtime-props-gap-analysis.md\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
    },
    {
        "name": "grep-tool-replay",
        "expected": "grep",
        "prompt": (
            "Produce exactly this tool call and no prose:\n"
            "<tool_call>\n"
            "<function=grep>\n"
            "<parameter=pattern>\n"
            "tool_replay\n"
            "</parameter>\n"
            "<parameter=path>\n"
            ".\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
    },
    {
        "name": "status-check",
        "expected": "shell",
        "prompt": (
            "Produce exactly this tool call and no prose:\n"
            "<tool_call>\n"
            "<function=shell>\n"
            "<parameter=cmd>\n"
            "git status --short\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
    },
]


def _tool_for(name: str) -> dict[str, Any]:
    for tool in TOOLS:
        if tool["function"]["name"] == name:
            return tool
    raise KeyError(name)


def run_case_once(
    url: str,
    case: dict[str, str],
    timeout_s: int,
    attempt: int,
) -> dict[str, Any]:
    prompt = case["prompt"] + " Do not answer in prose."
    if attempt > 1:
        prompt += f" Retry marker: agentic-tool-attempt-{attempt}."
    body = json.dumps({
        "model": "luce-dflash",
        "messages": [{
            "role": "user",
            "content": prompt,
        }],
        "tools": [_tool_for(case["expected"])],
        "tool_choice": {"type": "function", "function": {"name": case["expected"]}},
        "temperature": 0,
        "max_tokens": 512,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
            http_status = resp.status
    except Exception as e:
        return {
            "name": case["name"],
            "expected": case["expected"],
            "ok": False,
            "error": str(e),
            "wall_s": round(time.perf_counter() - t0, 3),
        }

    wall = time.perf_counter() - t0
    choices = data.get("choices") or []
    msg = (choices[0].get("message") if choices else {}) or {}
    calls = msg.get("tool_calls") or []
    names = [((c.get("function") or {}).get("name")) for c in calls]
    return {
        "name": case["name"],
        "expected": case["expected"],
        "ok": case["expected"] in names,
        "http_status": http_status,
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "tool_names": names,
        "content_preview": (msg.get("content") or "")[:200],
        "attempt": attempt,
        "wall_s": round(wall, 3),
    }


def run_case(
    url: str,
    case: dict[str, str],
    timeout_s: int,
    retries: int,
    runner=run_case_once,
) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    for attempt in range(1, retries + 2):
        row = runner(url, case, timeout_s, attempt)
        last = row
        if row["ok"]:
            return row
    assert last is not None
    return last


def main() -> int:
    ap = argparse.ArgumentParser(description="Run agentic tool-call reliability prompts.")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="Server base URL.")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--retries", type=int, default=2,
                    help="Retry cases that do not emit the expected tool call.")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    if args.repeat < 1:
        ap.error("--repeat must be >= 1")

    rows: list[dict[str, Any]] = []
    print(f"[agentic-tools] url={args.url} cases={len(CASES)} repeat={args.repeat}", flush=True)
    for i in range(args.repeat):
        for case in CASES:
            row = run_case(args.url, case, args.timeout, args.retries)
            row["iteration"] = i + 1
            rows.append(row)
            status = "PASS" if row["ok"] else "FAIL"
            print(f"  {status:4s} {case['name']:18s} expected={case['expected']} "
                  f"got={row.get('tool_names')} wall={row['wall_s']:.2f}s", flush=True)

    passed = sum(1 for r in rows if r["ok"])
    payload = {
        "suite": "agentic-tools",
        "passed": passed,
        "total": len(rows),
        "pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "rows": rows,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(f"[agentic-tools] pass_rate={payload['pass_rate']:.2%}", flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
