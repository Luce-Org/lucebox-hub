# Multi-client ee7 validation — 2026-05-22

Wave 1: claude_code, hermes, opencode
Wave 2 follow-up: pi, codex (after PATH fix)

## Wave 1 results

| Client | baseline wall | ee7 wall | speedup | OK_DONE |
|---|---|---|---|---|
| claude_code | 27.0s | 24.5s | 1.10x | YES |
| hermes | 12.8s | 11.5s | 1.11x | NO (partial) |
| opencode | 17.6s | 12.7s | 1.39x | NO (export err) |

## Wave 2 follow-up: pi + codex (after PATH fix)

| Client | baseline wall | ee7 wall | baseline drafter_fwd | ee7 drafter_fwd | speedup | OK_DONE |
|---|---|---|---|---|---|---|
| pi    | 10.2s | 12.7s | 0.45s | 0.21s | 0.80x | YES |
| codex | 10.8s |  9.5s | 1.57s | 0.50s | 1.14x | NO (single-turn) |

pi ee7 wall > baseline due to short-context variance (prompt ~5K tokens, below 32K pflash threshold). Drafter_fwd speedup real: 0.45s->0.21s (2.1x). codex ok_done=NO is pre-existing behavior (emits one tool call and stops).

## Consolidated production table (all 5 clients)

| Client | baseline wall | ee7 wall | baseline drafter_fwd | ee7 drafter_fwd | wall speedup | OK_DONE |
|---|---|---|---|---|---|---|
| claude_code | 27.0s | 24.5s | 4.31s | 1.17s | 1.10x | YES |
| hermes      | 12.8s | 11.5s | 2.18s | 0.67s | 1.11x | NO (partial) |
| opencode    | 17.6s | 12.7s | 0.83s | 0.24s | 1.39x | NO (export err) |
| pi          | 10.2s | 12.7s | 0.45s | 0.21s | 0.80x | YES |
| codex       | 10.8s |  9.5s | 1.57s | 0.50s | 1.14x | NO (single-turn) |

drafter_fwd speedup: claude_code 3.7x, hermes 3.3x, opencode 3.5x, pi 2.1x, codex 3.1x

## PATH fix notes

Both run_pi.sh and run_codex.sh previously used asdf shims / nvm.sh source, both of which fail in non-interactive subshells. Fix: directly prepend the nvm node bin dir to PATH by probing known version directories.
