"""Multi-turn coding-agent benchmark for lucebox HTTP servers.

This complements bench_agentic_tools.py. The tools benchmark proves small
single-turn OpenAI-format tool calls work. This benchmark replays the Anthropic
Messages wire shape used by Claude Code: tool_use blocks are returned by the
assistant, deterministic tool_result blocks are appended by the harness, and
context grows across turns.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = [
    {
        "type": "text",
        "text": "You are Claude Code compatibility smoke-test traffic.",
    },
    {
        "type": "text",
        "text": (
            "You are an interactive coding agent. Each user turn in this benchmark names "
            "one required tool. Emit exactly one tool_use block for that tool before any "
            "prose. Do not summarize until after a tool_result has been provided for the "
            "current turn."
        ),
    },
]

TOOLS = [
    {
        "name": "Read",
        "description": "Read a UTF-8 text file.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "Grep",
        "description": "Search files for a literal pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern", "path"],
        },
    },
    {
        "name": "Bash",
        "description": "Run a read-only shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]


def _block(label: str, repeat: int) -> str:
    line = (
        f"{label}: dflash server validates /props, OpenAI tool calls, "
        "frontier prompts, CUDA memory headroom, and benchmark snapshots.\n"
    )
    return line * repeat


DEFAULT_FIXTURE = [
    {
        "name": "repo-overview",
        "expected_tool": "Read",
        "user_msg": "Call the Read tool for README.md now. Emit no prose before tool_use.",
        "tool_result": "README.md\ndflash/scripts\nlucebox/lucebox\nlucebox/tests\n",
    },
    {
        "name": "props-search",
        "expected_tool": "Grep",
        "user_msg": (
            "Call the Grep tool to find where /props compatibility is implemented. "
            "Emit no prose before tool_use."
        ),
        "tool_result": _block("server.py grep /props", 20),
    },
    {
        "name": "bench-read",
        "expected_tool": "Read",
        "user_msg": (
            "Call the Read tool on dflash/scripts/lucebox_bench.py. "
            "Emit no prose before tool_use."
        ),
        "tool_result": _block("lucebox_bench.py", 45),
    },
    {
        "name": "agentic-tools-read",
        "expected_tool": "Read",
        "user_msg": (
            "Call the Read tool on dflash/scripts/bench_agentic_tools.py. "
            "Emit no prose before tool_use."
        ),
        "tool_result": _block("bench_agentic_tools.py", 70),
    },
    {
        "name": "quality-trace",
        "expected_tool": "Read",
        "user_msg": (
            "Call the Read tool on the latest ds4-eval trace. "
            "Emit no prose before tool_use."
        ),
        "tool_result": _block("bench-ds4-eval-trace.txt", 110),
    },
    {
        "name": "frontier-report",
        "expected_tool": "Read",
        "user_msg": (
            "Call the Read tool on bench-http-frontiers.json. "
            "Emit no prose before tool_use."
        ),
        "tool_result": _block("bench-http-frontiers.json", 160),
    },
    {
        "name": "snapshot-export",
        "expected_tool": "Read",
        "user_msg": (
            "Call the Read tool on lucebox/lucebox/profile.py. "
            "Emit no prose before tool_use."
        ),
        "tool_result": _block("profile.py", 220),
    },
    {
        "name": "final-plan",
        "expected_tool": "Bash",
        "user_msg": (
            "Call the Bash tool to inspect git status and dflash/scripts. "
            "Emit no prose before tool_use."
        ),
        "tool_result": _block("combined-session-context", 300),
    },
]


def fixture_hash(fixture: list[dict[str, str]]) -> str:
    import hashlib

    payload = json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _clean_text(value: str) -> str:
    return value.encode("utf-8", errors="replace").decode("utf-8")


def _anthropic_user_text(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _tool_result_message(tool_results: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": _clean_text(content),
            }
            for tool_use_id, content in tool_results
        ],
    }


def _message_chars(messages: list[dict[str, Any]]) -> int:
    return len(json.dumps(messages, sort_keys=True, separators=(",", ":")))


def _parse_anthropic_sse(
    resp,
    *,
    start_s: float,
) -> tuple[list[dict[str, Any]], dict[str, int], str | None, float | None]:
    blocks: dict[int, dict[str, Any]] = {}
    usage: dict[str, int] = {}
    stop_reason: str | None = None
    first_content_s: float | None = None
    event = ""
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
            continue
        if not line.startswith("data:"):
            continue
        payload = line.split(":", 1)[1].strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if data.get("usage"):
            usage.update(data["usage"])
        message = data.get("message") or {}
        if isinstance(message, dict) and message.get("usage"):
            usage.update(message["usage"])
        if event == "content_block_start":
            if first_content_s is None:
                first_content_s = time.perf_counter() - start_s
            idx = int(data.get("index", 0))
            blocks[idx] = data.get("content_block") or {}
        elif event == "content_block_delta":
            if first_content_s is None:
                first_content_s = time.perf_counter() - start_s
            idx = int(data.get("index", 0))
            block = blocks.setdefault(idx, {})
            delta = data.get("delta") or {}
            match delta.get("type"):
                case "text_delta":
                    block["text"] = str(block.get("text") or "") + str(delta.get("text") or "")
                case "thinking_delta":
                    block["thinking"] = (
                        str(block.get("thinking") or "") + str(delta.get("thinking") or "")
                    )
                case "input_json_delta":
                    block["_partial_json"] = (
                        str(block.get("_partial_json") or "")
                        + str(delta.get("partial_json") or "")
                    )
        elif event == "message_delta":
            delta = data.get("delta") or {}
            stop_reason = delta.get("stop_reason") or stop_reason
    out = []
    for idx in sorted(blocks):
        block = blocks[idx]
        if block.get("type") == "tool_use":
            partial = str(block.pop("_partial_json", "") or "")
            if partial:
                try:
                    block["input"] = json.loads(partial)
                except json.JSONDecodeError:
                    block["input"] = {}
        out.append(block)
    return out, usage, stop_reason, first_content_s


def run_turn(
    url: str,
    model: str,
    messages: list[dict[str, Any]],
    fixture_turn: dict[str, str],
    *,
    session_id: int,
    turn_idx: int,
    timeout_s: int,
    max_tokens: int,
) -> dict[str, Any]:
    messages.append(_anthropic_user_text(fixture_turn["user_msg"]))
    expected_tool = fixture_turn["expected_tool"]
    history_chars = _message_chars(messages)
    body = json.dumps({
        "model": model,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": {"type": "tool", "name": expected_tool},
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        url + "/v1/messages?beta=true",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        content, usage, stop_reason, first_content_s = _parse_anthropic_sse(resp, start_s=t0)

    wall = time.perf_counter() - t0
    tool_calls = [block for block in content if block.get("type") == "tool_use"]
    if not tool_calls:
        preview = json.dumps(content, sort_keys=True)[:500]
        raise RuntimeError(
            f"server returned no Anthropic tool_use for expected {expected_tool!r} "
            f"(session={session_id} turn={turn_idx + 1}); content={preview}"
        )
    tool_names = [str(call.get("name") or "") for call in tool_calls]
    if expected_tool not in tool_names:
        raise RuntimeError(
            f"server returned tool calls {tool_names} without expected {expected_tool!r} "
            f"(session={session_id} turn={turn_idx + 1})"
        )

    messages.append({"role": "assistant", "content": content})
    tool_results = [
        (str(call["id"]), fixture_turn["tool_result"])
        for call in tool_calls
    ]
    messages.append(_tool_result_message(tool_results))
    messages.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Tool result received. Ready for the next repository instruction.",
            }
        ],
    })

    completion_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    first_content_ms = round((first_content_s if first_content_s is not None else wall) * 1000, 2)
    return {
        "session": session_id,
        "turn": turn_idx + 1,
        "name": fixture_turn.get("name", f"turn-{turn_idx + 1}"),
        "ok": True,
        "expected_tool": expected_tool,
        "protocol": "anthropic_messages",
        "timing_mode": "stream_aggregated_content",
        "prompt_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
        "history_chars": history_chars,
        "request_bytes": len(body),
        "completion_tokens": completion_tokens,
        "stop_reason": stop_reason,
        "first_content_ms": first_content_ms,
        "wall_ms": round(wall * 1000, 2),
        "decode_tps": round(completion_tokens / wall, 2) if completion_tokens and wall > 0 else 0.0,
        "tool_calls": len(tool_calls),
        "tool_names": tool_names,
        "result_chars": len(fixture_turn["tool_result"]),
    }


def summarize(rows: list[dict[str, Any]], turns: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    first_wall = None
    for turn in range(1, turns + 1):
        group = [row for row in rows if row.get("turn") == turn and row.get("ok")]
        if not group:
            continue
        first_content = [
            float(row.get("first_content_ms") or row.get("wall_ms") or 0)
            for row in group
        ]
        walls = [float(row.get("wall_ms") or 0) for row in group]
        decode = [float(row["decode_tps"]) for row in group]
        prompt_token_values = [
            int(row["prompt_tokens"])
            for row in group
            if row.get("prompt_tokens") is not None
        ]
        history_chars = [int(row.get("history_chars") or 0) for row in group]
        request_bytes = [int(row.get("request_bytes") or 0) for row in group]
        if first_wall is None:
            first_wall = statistics.mean(walls)
        mean_wall = statistics.mean(walls)
        out.append({
            "turn": turn,
            "samples": len(group),
            "prompt_tokens_mean": (
                round(statistics.mean(prompt_token_values), 2)
                if prompt_token_values
                else None
            ),
            "history_chars_mean": round(statistics.mean(history_chars), 2),
            "request_bytes_mean": round(statistics.mean(request_bytes), 2),
            "first_content_ms_mean": round(statistics.mean(first_content), 2),
            "first_content_ms_std": (
                round(statistics.stdev(first_content), 2) if len(first_content) > 1 else 0.0
            ),
            "wall_ms_mean": round(mean_wall, 2),
            "wall_ms_std": round(statistics.stdev(walls), 2) if len(walls) > 1 else 0.0,
            "wall_growth_vs_turn1": round(mean_wall / first_wall, 3) if first_wall else 1.0,
            "decode_tps_mean": round(statistics.mean(decode), 2) if decode else 0.0,
            "tool_calls_mean": round(statistics.mean(row["tool_calls"] for row in group), 2),
            "result_chars_mean": round(statistics.mean(row["result_chars"] for row in group), 2),
        })
    return out


def run_session_bench(
    url: str,
    model: str,
    fixture: list[dict[str, str]],
    *,
    sessions: int,
    turns: int,
    timeout_s: int,
    max_tokens: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failed = False
    selected = fixture[:turns]
    for session_id in range(1, sessions + 1):
        messages: list[dict[str, Any]] = []
        for turn_idx, turn in enumerate(selected):
            try:
                row = run_turn(
                    url, model, messages, turn,
                    session_id=session_id, turn_idx=turn_idx,
                    timeout_s=timeout_s, max_tokens=max_tokens,
                )
            except Exception as e:
                row = {
                    "session": session_id,
                    "turn": turn_idx + 1,
                    "name": turn.get("name", f"turn-{turn_idx + 1}"),
                    "ok": False,
                    "error": str(e),
                    "result_chars": len(turn["tool_result"]),
                    "history_chars": _message_chars(messages),
                }
                failed = True
            rows.append(row)
            if not row["ok"]:
                break
    summary = summarize(rows, turns)
    passed = sum(1 for row in rows if row.get("ok"))
    total = sessions * len(selected)
    final = summary[-1] if summary else {}
    return {
        "suite": "agentic-session",
        "source": "club-3090-inspired-claude-code-anthropic-messages",
        "fixture_id": "lucebox-agentic-session-v1",
        "fixture_hash": fixture_hash(selected),
        "sessions": sessions,
        "turns": len(selected),
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "ok": not failed and passed == total,
        "max_prompt_tokens": max(
            (
                int(row["prompt_tokens"])
                for row in rows
                if row.get("prompt_tokens") is not None
            ),
            default=None,
        ),
        "final_wall_growth_vs_turn1": final.get("wall_growth_vs_turn1"),
        "summary": summary,
        "rows": rows,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run multi-turn agentic session benchmark.")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="luce-dflash")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    turns = max(1, min(args.turns, len(DEFAULT_FIXTURE)))
    sessions = max(1, args.sessions)
    print(
        f"[agentic-session] url={args.url} sessions={sessions} turns={turns}",
        flush=True,
    )
    payload = run_session_bench(
        args.url, args.model, DEFAULT_FIXTURE,
        sessions=sessions, turns=turns,
        timeout_s=args.timeout, max_tokens=args.max_tokens,
    )
    for row in payload["summary"]:
        prompt_tokens = row["prompt_tokens_mean"]
        prompt_text = "unknown" if prompt_tokens is None else f"{prompt_tokens:.0f}"
        print(
            f"  turn={row['turn']:02d} prompt_tok={prompt_text} "
            f"first_content={row['first_content_ms_mean']:.0f}ms "
            f"wall={row['wall_ms_mean']:.0f}ms "
            f"growth={row['wall_growth_vs_turn1']:.2f}x "
            f"decode={row['decode_tps_mean']:.1f} tok/s",
            flush=True,
        )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
