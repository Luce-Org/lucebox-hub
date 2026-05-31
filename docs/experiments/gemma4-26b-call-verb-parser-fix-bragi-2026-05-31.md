# Gemma4 26B call:<verb>{} parser fix — bragi — 2026-05-31

Verification that the server-side `call:<verb>{}` tool parser (PR #323,
merged via easel into `feat/lucebox-docker`) fixes Gemma4's forge=0% issue.

* **Host**: bragi (RTX 5090 Laptop MaxQ, 23 GB VRAM, WSL2, sm_120)
  * GPU throttled to ~86–90 W / 1515 MHz (Windows Balanced mode).
* **Image**: locally-built `lucebox-hub:cuda12` @ `14432393`
  * Built with `DFLASH_CUDA_ARCHES=120` for sm_120 (Blackwell).
  * Includes call:<verb>{} parser (commit `5ca695cd` / PR #323).
  * Also includes C++17 compat fix for `starts_with` → `rfind` (commit
    `14432393`; `std::string::starts_with` is C++20 but CMakeLists.txt
    requires C++17).
* **Server config** (Gemma4 optimal from 2026-05-30 bragi sweep):
  ```toml
  budget = 22
  max_ctx = 131072
  cache_type_k = "tq3_0"
  cache_type_v = "tq3_0"
  fa_window = 0
  think_max = 15488
  prefix_cache_slots = 0
  prefill_cache_slots = 0
  ```
  Note: Gemma4 KV quantization is hardcoded F16 in gemma4_loader.cpp;
  cache_type_k/v settings have no effect.

## Background: why forge=0%?

Gemma4 emits tool calls in a plain-text format: `call:<verb>{<json-args>}`.
Example:
```
call:get_file{"path": "src/main.py"}
```

Before PR #323, luce-bench's client-side workaround parsed this format in
Python. The server's `tool_parser.cpp` only had patterns for:
- `<tool_call>...</tool_call>` (XML)
- `{"name": "...", "parameters": {...}}` (JSON)

Since the server couldn't parse `call:<verb>{}`, it returned these strings
as raw `content` text, not as `tool_calls`. luce-bench's client workaround
was supposed to catch this, but forge uses a different code path from
agent_recorded and didn't benefit from the workaround → forge=0%.

PR #323 adds pattern #6 to tool_parser.cpp:
```cpp
static const std::regex & re_call_verb_open() {
    static std::regex r(R"((^|[\s,;:\(\[\{\}\)\]\>])call:([A-Za-z0-9_.:\-]+)\s*\{)");
    return r;
}
```
Uses `balanced_braces_end()` for nested JSON. This is the correct fix: the
server now returns proper `tool_calls` for Gemma4's emission format.

## forge benchmark: before vs after fix

`uv run luce-bench --areas forge --no-think`

| Leg | image | forge pass_rate | notes |
|-----|-------|-----------------|-------|
| pre-fix (baseline) | a45c9fa | ~0% | tool calls not parsed; all scenarios fail |
| post-fix (2026-05-31) | 14432393 | *TBD* | running |

## agent_recorded benchmark: Gemma4 vs Qwen3.6-27B

`uv run luce-bench --areas agent_recorded --no-think`

| Model | Config | Score | Notes |
|-------|--------|-------|-------|
| Qwen3.6-27B (Q4_K_M) | budget=16, max_ctx=98304, kv=tq3_0 | 46.2% (12/26) | 2026-05-31, tq3_0 KV |
| Gemma4-26B-A4B (Q4_K_M) | budget=22, max_ctx=131072, kv=F16 | *TBD* | running |

Note: Gemma4 decode rate is ~64-65 tok/s vs Qwen3.6 ~24 tok/s (2.7× faster
generation) because Gemma4-26B-A4B is a sparse MoE model (4B active params
per token vs 27B dense for Qwen3.6). Prefill is proportionally slower for
Gemma4 due to larger vocabulary (262144 vs 151936).

## C++17 compat fix note

The Gemma4 channel-token routing commit (`4b757d10`) introduced
`std::string::starts_with()` which requires C++20. The Docker image failed
to build until `14432393` replaced both instances with `rfind("<|channel>", 0) == 0`
(idiomatic C++17 equivalent). The CMakeLists.txt remains at C++17 standard
which is compatible with llama.cpp's own standard setting.

## Next steps

1. Wait for forge and agent_recorded benchmarks to complete
2. Fill in results table
3. Compare Gemma4 forge pre/post fix (expect significant improvement)
4. Compare Gemma4 vs Qwen3.6 on agent_recorded quality
5. Assess which model is better for the coding-agent-loop use case
