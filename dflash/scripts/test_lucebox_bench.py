import subprocess
import sys
from pathlib import Path

import bench_agentic_session
import bench_agentic_tools
import bench_he
import bench_http_capability
import bench_http_frontiers
import lucebox_bench


class StubProcess:
    pid = 123

    def wait(self, timeout=None):
        return 0


class StubExtraSuiteRuntime:
    def __init__(self, returncode=0, ready=True):
        self.returncode = returncode
        self.ready = ready
        self.calls = []
        self.stopped = False

    def build_server_argv(self, _cfg, _target):
        return ["server"]

    def start_server(self, argv):
        self.calls.append(("start", argv))
        return StubProcess()

    def wait_ready(self, timeout_s):
        self.calls.append(("wait_ready", timeout_s))
        return self.ready

    def call(self, cmd):
        self.calls.append(("call", cmd))
        return self.returncode

    def stop_server(self, _proc):
        self.stopped = True


def _cell(max_ctx, budget, mean, status="ok", min_tps=None):
    return {
        "max_ctx": max_ctx,
        "budget": budget,
        "status": status,
        "mean_decode_tps": mean,
        "min_decode_tps": mean if min_tps is None else min_tps,
    }


def _cfg(**kwargs):
    fields = {
        "max_ctx": 32768,
        "budget": 22,
        "lazy": True,
        "prefix_cache_slots": 0,
        "prefill_cache_slots": 0,
        "kv": "auto",
        "prefill_mode": "off",
        "prefill_keep_ratio": 0.05,
        "prefill_threshold": 32000,
        "prefill_drafter": "",
    }
    fields.update(kwargs)
    return lucebox_bench.SweepConfig(**fields)


def test_capability_help_renders_percent_literal():
    result = subprocess.run(
        [sys.executable, str(Path(bench_http_capability.__file__)), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "%%" not in result.stdout


def test_find_target_gguf_prefers_largest_qwen_match(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    small = models / "a-Qwen3.6-small-Q4_K_M.gguf"
    small.write_bytes(b"small")
    large = models / "z-Qwen3.6-large-Q4_K_M.gguf"
    with large.open("wb") as f:
        f.truncate(1024)
    other = models / "other.gguf"
    with other.open("wb") as f:
        f.truncate(4096)

    monkeypatch.setattr(lucebox_bench, "MODELS_DIR", models)
    monkeypatch.delenv("DFLASH_TARGET", raising=False)

    assert lucebox_bench.find_target_gguf() == large


def test_pick_winner_level1_prefers_speed():
    winner = lucebox_bench.pick_winner([
        _cell(131072, 22, 18.0),
        _cell(32768, 16, 25.0),
    ], profile="level1")

    assert winner["max_ctx"] == 32768
    assert winner["budget"] == 16


def test_pick_winner_level2_prefers_highest_context_within_speed_floor():
    winner = lucebox_bench.pick_winner([
        _cell(32768, 16, 25.0),
        _cell(65536, 22, 23.0),
        _cell(131072, 22, 18.0),
    ], profile="level2", min_context_speed_ratio=0.85)

    assert winner["max_ctx"] == 65536
    assert winner["budget"] == 22


def test_rank_candidates_level2_keeps_fallback_order():
    ranked = lucebox_bench.rank_candidates([
        _cell(32768, 16, 30.0),
        _cell(98304, 16, 20.0),
        _cell(98304, 22, 26.0),
        _cell(114688, 22, 23.0),
    ], profile="level2", min_context_speed_ratio=0.75)

    assert [(c["max_ctx"], c["budget"]) for c in ranked[:3]] == [
        (114688, 22),
        (98304, 22),
        (32768, 16),
    ]


def test_rank_candidates_level3_uses_context_policy():
    ranked = lucebox_bench.rank_candidates([
        _cell(65536, 16, 26.0),
        _cell(98304, 16, 24.0),
        _cell(114688, 22, 18.0),
    ], profile="level3", min_context_speed_ratio=0.85)

    assert [(c["max_ctx"], c["budget"]) for c in ranked] == [
        (98304, 16),
        (65536, 16),
    ]


def test_rank_candidates_level_profiles_use_expected_policies():
    level1 = lucebox_bench.rank_candidates([
        _cell(32768, 16, 25.0),
        _cell(65536, 22, 24.0),
    ], profile="level1")
    level2 = lucebox_bench.rank_candidates([
        _cell(32768, 16, 25.0),
        _cell(65536, 22, 24.0),
    ], profile="level2", min_context_speed_ratio=0.85)

    assert level1[0]["max_ctx"] == 32768
    assert level2[0]["max_ctx"] == 65536


def test_pick_winner_level2_uses_fastest_budget_inside_highest_viable_context():
    winner = lucebox_bench.pick_winner([
        _cell(32768, 16, 25.0),
        _cell(65536, 16, 22.0),
        _cell(65536, 22, 23.0),
    ], profile="level2", min_context_speed_ratio=0.85)

    assert winner["max_ctx"] == 65536
    assert winner["budget"] == 22


def test_context_values_for_profile_level1_uses_configured_env():
    assert lucebox_bench.context_values_for_profile(
        "level1", "", {"DFLASH_MAX_CTX": "114688"}) == [114688]


def test_context_values_for_profile_level2_caps_to_configured_env():
    assert lucebox_bench.context_values_for_profile(
        "level2", "", {"DFLASH_MAX_CTX": "98304"}) == [
        32768, 65536, 98304,
    ]


def test_level_profiles_and_prompt_sets_are_explicit():
    env = {"DFLASH_MAX_CTX": "98304"}
    assert lucebox_bench.normalize_profile("level1") == "level1"
    assert lucebox_bench.normalize_profile("level2") == "level2"
    assert lucebox_bench.normalize_profile("level3") == "level3"
    assert lucebox_bench.context_values_for_profile("level1", "", env) == [98304]
    assert lucebox_bench.context_values_for_profile("level2", "", env) == [
        32768, 65536, 98304,
    ]
    assert lucebox_bench.default_extra_suites_for_profile("level1") == [
        "capability", "agentic-tools", "http-frontiers", "agentic-session",
    ]
    # agentic-session runs in every profile so snapshots from different
    # machines have matching section sets to diff against. Profiles still
    # differ on sweep size and on whether the ds4-eval/long capability gates
    # run, just not on whether agentic-session runs.
    assert "agentic-tools" in lucebox_bench.default_extra_suites_for_profile("level2")
    assert "agentic-session" in lucebox_bench.default_extra_suites_for_profile("level2")
    assert "agentic-session" in lucebox_bench.default_extra_suites_for_profile("level3")
    assert bench_he.PROMPTS[0][0] == "has_close_elements"
    assert lucebox_bench.select_prompts(1)[0][0] == "scheduler_run_spec"


def test_sweep_configs_expands_requested_tunables():
    args = type("Args", (), {
        "profile": "level2",
        "ctx_values": "32768",
        "budgets": "16,22",
        "lazy_values": "0,1",
        "prefix_cache_slots_values": "0,1",
        "prefill_cache_slots_values": "",
        "kv_values": "auto,q4_0",
        "prefill_modes": "off",
        "prefill_keep_ratios": "",
        "prefill_thresholds": "",
        "prefill_drafter": "",
    })()

    configs = lucebox_bench.sweep_configs(args, {"DFLASH_MAX_CTX": "65536"})

    assert len(configs) == 16
    assert any(c.budget == 16 and c.lazy is False and c.kv == "q4_0" for c in configs)


def test_sweep_configs_defaults_to_current_kv():
    args = type("Args", (), {
        "profile": "level1",
        "ctx_values": "32768",
        "budgets": "22",
        "lazy_values": "",
        "prefix_cache_slots_values": "",
        "prefill_cache_slots_values": "",
        "kv_values": "",
        "prefill_modes": "off",
        "prefill_keep_ratios": "",
        "prefill_thresholds": "",
        "prefill_drafter": "",
    })()

    configs = lucebox_bench.sweep_configs(args, {
        "DFLASH_MAX_CTX": "65536",
        "DFLASH_CACHE_TYPE_K": "q4_0",
        "DFLASH_CACHE_TYPE_V": "q4_0",
    })

    assert [c.kv for c in configs] == ["q4_0"]


def test_build_server_argv_includes_tuned_flags(tmp_path):
    target = tmp_path / "target.gguf"
    target.write_text("")
    drafter = tmp_path / "Qwen3-0.6B-BF16.gguf"
    drafter.write_text("")
    cfg = _cfg(
        kv="q4_0",
        prefix_cache_slots=0,
        prefill_cache_slots=1,
        prefill_mode="auto",
        prefill_threshold=512,
        prefill_keep_ratio=0.1,
        prefill_drafter=str(drafter),
    )

    argv = lucebox_bench.build_server_argv(cfg, target)

    assert "--cache-type-k" in argv
    assert "--cache-type-v" in argv
    assert "--prefix-cache-slots" in argv
    assert argv[argv.index("--prefix-cache-slots") + 1] == "0"
    assert "--prefill-cache-slots" in argv
    assert argv[argv.index("--prefill-cache-slots") + 1] == "1"
    assert "--prefill-compression" in argv
    assert argv[argv.index("--prefill-compression") + 1] == "auto"


def test_build_server_argv_persists_f16_as_cache_type(tmp_path):
    target = tmp_path / "target.gguf"
    target.write_text("")

    argv = lucebox_bench.build_server_argv(_cfg(kv="f16"), target)

    assert "--kv-f16" not in argv
    assert argv[argv.index("--cache-type-k") + 1] == "f16"
    assert argv[argv.index("--cache-type-v") + 1] == "f16"


def test_http_frontier_retries_empty_completion():
    rows = [
        {"completion_tokens": 0},
        {"completion_tokens": 12},
    ]

    def fake_run_one(*_args):
        return rows.pop(0)

    row = bench_http_frontiers.run_frontier(
        "http://test", "prompt", 16, 1, retries=2, runner=fake_run_one)

    assert row["completion_tokens"] == 12
    assert row["attempt"] == 2


def test_http_frontiers_rejects_empty_frontier_list():
    result = subprocess.run(
        [
            sys.executable,
            str(Path(bench_http_frontiers.__file__)),
            "--frontiers",
            "",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--frontiers must include at least one integer" in result.stderr


def test_agentic_tools_rejects_nonpositive_repeat():
    result = subprocess.run(
        [
            sys.executable,
            str(Path(bench_agentic_tools.__file__)),
            "--repeat",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--repeat must be >= 1" in result.stderr


def test_capability_answer_extractors_ignore_hidden_thinking():
    assert bench_http_capability.find_answer(
        {"kind": "choice", "choices": ["x", "y"], "answer": "B"},
        "<think>Answer: A</think>\nAnswer: B",
    ) == "B"
    assert bench_http_capability.find_answer(
        {"kind": "integer", "answer": "42"},
        "scratch 7\nAnswer: 0042",
    ) == "42"


def test_capability_answer_extractors_require_final_answer_line():
    assert bench_http_capability.find_answer(
        {"kind": "choice", "choices": ["x", "y"], "answer": "B"},
        "The answer is probably B.",
    ) == "?"
    assert bench_http_capability.find_answer(
        {"kind": "integer", "answer": "42"},
        "scratch 7 then 42",
    ) == "?"


def test_capability_semantic_hint_does_not_weaken_smoke_grading():
    case = {"area": "smoke", "kind": "integer", "answer": "42"}

    assert bench_http_capability.find_answer(case, "scratch 7 then 42") == "?"
    assert bench_http_capability.semantic_hint_present(case, "scratch 7 then 42", "") is True
    grade = bench_http_capability.grade_case(case, "?", "scratch 7 then 42", "")
    assert grade["ok"] is False
    assert grade["status"] == "format_error"


def test_capability_semantic_pass_requires_explicit_score_mode():
    case = {"area": "diagnostic", "kind": "integer", "answer": "42", "score_mode": "semantic"}

    grade = bench_http_capability.grade_case(case, "?", "scratch 7 then 42", "")

    assert grade["ok"] is True
    assert grade["strict_pass"] is False
    assert grade["format_pass"] is False
    assert grade["semantic_pass"] is True
    assert grade["status"] == "semantic_pass"


def test_capability_semantic_hint_handles_multi_answer_line_cases():
    case = {"area": "diagnostic", "kind": "line", "answer": ["3", "4"]}

    assert bench_http_capability.semantic_hint_present(
        case,
        "The check on line 3 is insufficient, but line 4 performs the copy.",
        None,
    ) is True


def test_capability_case_selection_and_multi_answer_grading():
    smoke = bench_http_capability.select_cases("smoke")
    ds4_eval = bench_http_capability.select_cases("ds4-eval")

    assert [case["id"] for case in smoke] == ["arithmetic-choice", "short-recall"]
    assert len(ds4_eval) == 92
    assert ds4_eval[0]["id"] == "recNu3MXkvWUzHZr9"
    assert ds4_eval[0]["source"] == "GPQA Diamond"
    assert ds4_eval[-1]["id"] == "compsec-092"
    assert bench_http_capability.expected_answers(ds4_eval[-1]) == ["10-14"]
    assert [case["id"] for case in bench_http_capability.select_cases(
        "ds4-eval", case_index=1)] == ["recNu3MXkvWUzHZr9"]
    assert [case["ds4_index"] for case in bench_http_capability.select_cases(
        "ds4-eval", case_id="001b51d76b4d422988f2c11f104a2c6c")] == [2]


def test_capability_default_gating_matches_documented_areas():
    assert bench_http_capability.default_min_pass_rate("smoke") == 1.0
    assert bench_http_capability.default_min_pass_rate("long") == 0.0
    assert bench_http_capability.default_min_pass_rate("ds4-eval") == 0.0


def test_capability_ds4_defaults_match_upstream_budget():
    # antirez/ds4 ds4_eval.c `.max_tokens = 16000` — the combined cap
    # covering reasoning + reply. The thinking budget split is server
    # config (`--think-max-tokens` on dflash), not wire protocol, so the
    # bench just sends a standard OpenAI max_tokens and lets each server
    # apply its own policy.
    assert bench_http_capability.DS4_EVAL_MAX_TOKENS == 16000
    assert bench_http_capability.default_max_tokens("smoke") == 512
    assert bench_http_capability.default_max_tokens("long") == 512
    assert bench_http_capability.default_max_tokens("ds4-eval") == 16000
    assert bench_http_capability.default_thinking_enabled("smoke") is False
    assert bench_http_capability.default_thinking_enabled("ds4-eval") is True


def test_capability_ds4_extraction_matches_compsec_range_grading():
    case = {
        "area": "ds4-eval",
        "source": "COMPSEC",
        "id": "compsec-test",
        "kind": "compsec",
        "answer": "10-14",
        "ds4_eval": True,
    }

    got = bench_http_capability.find_answer(case, "Analysis...\nAnswer: line 12")
    grade = bench_http_capability.grade_case(case, got, "Analysis...\nAnswer: line 12", "")

    assert got == "12"
    assert grade["strict_pass"] is True
    assert grade["semantic_pass"] is False


def test_capability_ds4_does_not_grade_hidden_reasoning_as_answer():
    case = {
        "area": "ds4-eval",
        "source": "AIME2025",
        "id": "aime-test",
        "kind": "integer",
        "answer": "42",
        "ds4_eval": True,
    }

    got = bench_http_capability.find_answer_with_fallback(
        case, "", "Reasoning arrives at Answer: 42"
    )
    grade = bench_http_capability.grade_case(case, got, "", "Reasoning arrives at Answer: 42")

    assert got == "?"
    assert grade["ok"] is False
    assert grade["semantic_hint"] is True


def test_agentic_tool_case_retries_until_expected_call():
    rows = [
        {"ok": False, "attempt": 1},
        {"ok": True, "attempt": 2},
    ]

    def fake_run_case_once(*_args):
        return rows.pop(0)

    row = bench_agentic_tools.run_case(
        "http://test", {"name": "x", "expected": "read_file"}, timeout_s=1, retries=2,
        runner=fake_run_case_once)

    assert row["ok"] is True
    assert row["attempt"] == 2


def test_agentic_session_summary_tracks_context_growth():
    rows = [
        {
            "turn": 1,
            "ok": True,
            "prompt_tokens": 100,
            "first_content_ms": 45,
            "wall_ms": 50,
            "decode_tps": 25,
            "tool_calls": 1,
            "result_chars": 10,
        },
        {
            "turn": 2,
            "ok": True,
            "prompt_tokens": 400,
            "first_content_ms": 115,
            "wall_ms": 125,
            "decode_tps": 20,
            "tool_calls": 1,
            "result_chars": 100,
        },
    ]

    summary = bench_agentic_session.summarize(rows, turns=2)

    assert summary[0]["wall_growth_vs_turn1"] == 1.0
    assert summary[1]["prompt_tokens_mean"] == 400
    assert summary[1]["first_content_ms_mean"] == 115
    assert summary[1]["wall_growth_vs_turn1"] == 2.5
    assert bench_agentic_session.fixture_hash(bench_agentic_session.DEFAULT_FIXTURE[:2])


def test_run_extra_suites_marks_failed_suite_and_cleans_process(tmp_path):
    runtime = StubExtraSuiteRuntime(returncode=1)

    results = lucebox_bench.run_extra_suites(
        _cfg(), tmp_path / "target.gguf", ["capability"], 1, "512", 1,
        runtime=runtime, report_dir=tmp_path)

    assert len(results) == 1
    assert results[0]["suite"] == "capability"
    assert results[0]["status"] == "failed"
    assert results[0]["returncode"] == 1
    assert results[0]["report"] == str(tmp_path / "bench-capability.json")
    assert runtime.stopped is True


def test_run_extra_suites_records_ds4_eval_without_hard_gate(tmp_path):
    runtime = StubExtraSuiteRuntime(returncode=0)

    results = lucebox_bench.run_extra_suites(
        _cfg(), tmp_path / "target.gguf", ["ds4-eval"], 1, "512", 1,
        runtime=runtime, report_dir=tmp_path)

    assert results[0]["suite"] == "ds4-eval"
    assert results[0]["status"] == "ok"


def test_run_extra_suites_uses_long_context_as_hard_gate(tmp_path):
    runtime = StubExtraSuiteRuntime(returncode=1)

    results = lucebox_bench.run_extra_suites(
        _cfg(), tmp_path / "target.gguf", ["capability-long"], 1, "512", 1,
        runtime=runtime, report_dir=tmp_path)

    assert results[0]["suite"] == "capability-long"
    assert results[0]["status"] == "failed"


def test_extra_suite_command_wires_agentic_session():
    suite = lucebox_bench.extra_suite_command(
        "agentic-session",
        frontiers="512",
        agentic_repeat=1,
        agentic_session_turns=4,
        agentic_session_sessions=2,
    )

    assert suite is not None
    suite_name, out_json, cmd = suite
    assert suite_name == "agentic-session"
    assert out_json.name == "bench-agentic-session.json"
    assert cmd[1].endswith("bench_agentic_session.py")
    assert cmd[cmd.index("--turns") + 1] == "4"
    assert cmd[cmd.index("--sessions") + 1] == "2"


def test_extra_suite_command_keeps_ds4_eval_score_only_gate():
    suite = lucebox_bench.extra_suite_command(
        "ds4-eval",
        frontiers="512",
        agentic_repeat=1,
    )

    assert suite is not None
    suite_name, out_json, cmd = suite
    assert suite_name == "ds4-eval"
    assert out_json.name == "bench-ds4-eval.json"
    assert "--area" in cmd
    assert cmd[cmd.index("--area") + 1] == "ds4-eval"
    assert "--max-tokens" in cmd
    assert cmd[cmd.index("--max-tokens") + 1] == "4096"
    assert "--think" in cmd
    assert "--min-pass-rate" in cmd
    assert cmd[cmd.index("--min-pass-rate") + 1] == "0.0"


def test_extra_suite_command_keeps_long_context_as_hard_gate():
    suite = lucebox_bench.extra_suite_command(
        "capability-long",
        frontiers="512",
        agentic_repeat=1,
    )

    assert suite is not None
    suite_name, _out_json, cmd = suite
    assert suite_name == "capability-long"
    assert "--area" in cmd
    assert cmd[cmd.index("--area") + 1] == "long"
    assert "--min-pass-rate" in cmd
    assert cmd[cmd.index("--min-pass-rate") + 1] == "1.0"


def test_entrypoint_shell_syntax_validates():
    result = subprocess.run(
        ["bash", "-n", str(Path(__file__).resolve().parent / "entrypoint.sh")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
