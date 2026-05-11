"""Tests for configlib.validate."""
import pytest
from dflash.scripts.configlib.validate import validate_profile, ProfileError


def _make_profile(overrides=None):
    p = {
        "extends": None,
        "backend": "dflash",
        "hardware": {"gpu": "RTX 3090", "sm": 86},
        "model": {"target": "/some/model.gguf"},
        "runtime": {
            "ctx": 4096,
            "kv_k": "q8_0",
            "kv_v": "q8_0",
            "spec": {"method": "none"},
        },
        "expected_floors": {"decode_tok_s": 5.0},
        "provenance": {
            "source_log": "some/existing/file.toml",
            "measured_at": "2026-01-01",
            "hardware_id": "test-device",
        },
    }
    if overrides:
        _deep_update(p, overrides)
    return p


def _deep_update(base, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict) and v:
            _deep_update(base[k], v)
        else:
            base[k] = v


def test_needs_run_source_log_is_warning(tmp_path):
    profile = _make_profile({"provenance": {"source_log": "<NEEDS_RUN>"}})
    errors, warnings = validate_profile(profile, profile_name="p.toml", strict=False)
    assert not errors
    assert any("NEEDS_RUN" in w for w in warnings)


def test_needs_run_with_strict_is_error():
    profile = _make_profile({"provenance": {"source_log": "<NEEDS_RUN>"}})
    errors, warnings = validate_profile(profile, profile_name="p.toml", strict=True)
    assert any("NEEDS_RUN" in e for e in errors)


def test_missing_provenance_is_error():
    profile = _make_profile()
    del profile["provenance"]
    errors, warnings = validate_profile(profile, profile_name="p.toml")
    assert any("provenance" in e.lower() for e in errors)


def test_empty_floors_is_error():
    profile = _make_profile({"expected_floors": {}})
    errors, warnings = validate_profile(profile, profile_name="p.toml")
    assert any("floor" in e.lower() or "expected_floors" in e.lower() for e in errors)


def test_mtp_without_assistant_is_error():
    profile = _make_profile({
        "runtime": {"spec": {"method": "mtp", "gamma": 2}},
    })
    errors, warnings = validate_profile(profile, profile_name="p.toml")
    assert any("mtp_assistant" in e.lower() or "assistant" in e.lower() for e in errors)


def test_dflash_without_draft_is_error():
    profile = _make_profile({
        "runtime": {"spec": {"method": "dflash", "draft_max": 4}},
    })
    errors, warnings = validate_profile(profile, profile_name="p.toml")
    assert any("dflash_draft" in e.lower() or "draft" in e.lower() for e in errors)


def test_valid_mtp_profile_no_errors():
    profile = _make_profile({
        "model": {"target": "/some/model.gguf", "mtp_assistant": "/some/assistant.gguf"},
        "runtime": {"spec": {"method": "mtp", "gamma": 2}},
    })
    errors, warnings = validate_profile(profile, profile_name="p.toml")
    assert not errors


def test_valid_dflash_profile_no_errors():
    profile = _make_profile({
        "model": {"target": "/some/model.gguf", "dflash_draft": "/some/draft.gguf"},
        "runtime": {"spec": {"method": "dflash", "draft_max": 4}},
    })
    errors, warnings = validate_profile(profile, profile_name="p.toml")
    assert not errors
