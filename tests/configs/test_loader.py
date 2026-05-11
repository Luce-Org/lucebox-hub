"""Tests for configlib.loader — TDD harness."""
import os
import pytest

from dflash.scripts.configlib.loader import load_profile, ProfileError


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return p


MINIMAL_TOML = (
    'extends = ""\n'
    'backend = "dflash"\n'
    '\n'
    '[hardware]\n'
    'gpu = "RTX 3090"\n'
    'sm = 86\n'
    '\n'
    '[model]\n'
    'target = "models/base.gguf"\n'
    '\n'
    '[runtime]\n'
    'ctx = 4096\n'
    'kv_k = "q8_0"\n'
    'kv_v = "q8_0"\n'
    '\n'
    '[runtime.spec]\n'
    'method = "none"\n'
    '\n'
    '[expected_floors]\n'
    'decode_tok_s = 5.0\n'
    '\n'
    '[provenance]\n'
    'source_log = "tests/configs/fixtures/base.toml"\n'
    'measured_at = "2026-01-01"\n'
    'hardware_id = "test-device"\n'
)


def test_valid_base_parses(tmp_path):
    p = _write(tmp_path, "base.toml", MINIMAL_TOML)
    profile = load_profile(p, git_root=str(tmp_path))
    assert profile["backend"] == "dflash"
    assert profile["hardware"]["gpu"] == "RTX 3090"
    assert profile["runtime"]["ctx"] == 4096


def test_child_inherits_and_overrides(tmp_path):
    _write(tmp_path, "base.toml", MINIMAL_TOML)
    child_toml = (
        MINIMAL_TOML
        .replace('extends = ""', 'extends = "base"')
        .replace('target = "models/base.gguf"', 'target = "models/child.gguf"')
        .replace("ctx = 4096", "ctx = 8192")
        .replace('kv_k = "q8_0"', 'kv_k = "tq3_0"')
        .replace('kv_v = "q8_0"', 'kv_v = "tq3_0"')
        .replace("decode_tok_s = 5.0", "decode_tok_s = 8.0")
    )
    p = _write(tmp_path, "child.toml", child_toml)
    profile = load_profile(p, git_root=str(tmp_path), profiles_dir=str(tmp_path))
    assert profile["runtime"]["ctx"] == 8192
    assert profile["runtime"]["kv_k"] == "tq3_0"
    assert profile["hardware"]["sm"] == 86


def test_missing_profile_file_raises(tmp_path):
    with pytest.raises(ProfileError, match="not found"):
        load_profile(tmp_path / "nonexistent.toml", git_root=str(tmp_path))


def test_toml_parse_error_raises(tmp_path):
    p = _write(tmp_path, "bad.toml", "this = [broken toml {{{")
    with pytest.raises(ProfileError, match="TOML"):
        load_profile(p, git_root=str(tmp_path))


def test_circular_extends_raises(tmp_path):
    a_toml = MINIMAL_TOML.replace('extends = ""', 'extends = "b"')
    b_toml = MINIMAL_TOML.replace('extends = ""', 'extends = "a"')
    _write(tmp_path, "a.toml", a_toml)
    _write(tmp_path, "b.toml", b_toml)
    with pytest.raises(ProfileError, match="[Cc]ircular"):
        load_profile(tmp_path / "a.toml", git_root=str(tmp_path), profiles_dir=str(tmp_path))


def test_env_var_expands(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_MODELS", str(tmp_path))
    toml = MINIMAL_TOML.replace(
        'target = "models/base.gguf"',
        'target = "${MY_MODELS}/models/base.gguf"'
    )
    p = _write(tmp_path, "profile.toml", toml)
    profile = load_profile(p, git_root=str(tmp_path))
    assert profile["model"]["target"] == f"{tmp_path}/models/base.gguf"


def test_env_var_default_used_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSET_VAR_XYZ", raising=False)
    toml = MINIMAL_TOML.replace(
        'target = "models/base.gguf"',
        'target = "${UNSET_VAR_XYZ:-models}/base.gguf"'
    )
    p = _write(tmp_path, "profile.toml", toml)
    profile = load_profile(p, git_root=str(tmp_path))
    assert profile["model"]["target"].endswith("models/base.gguf")


def test_unset_required_var_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("LUCEBOX_ROOT", raising=False)
    toml = MINIMAL_TOML.replace(
        'target = "models/base.gguf"',
        'target = "${LUCEBOX_ROOT}/models/base.gguf"'
    )
    p = _write(tmp_path, "profile.toml", toml)
    with pytest.raises(ProfileError) as exc_info:
        load_profile(p, git_root=str(tmp_path))
    err = str(exc_info.value)
    assert "LUCEBOX_ROOT" in err
    assert "profile.toml" in err


def test_hardcoded_absolute_path_raises(tmp_path):
    toml = MINIMAL_TOML.replace(
        'target = "models/base.gguf"',
        'target = "/absolute/path/model.gguf"'
    )
    p = _write(tmp_path, "profile.toml", toml)
    with pytest.raises(ProfileError, match="[Hh]ardcoded absolute"):
        load_profile(p, git_root=str(tmp_path))


def test_env_expanded_absolute_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_ROOT", "/some/absolute/root")
    toml = MINIMAL_TOML.replace(
        'target = "models/base.gguf"',
        'target = "${MY_ROOT}/models/base.gguf"'
    )
    p = _write(tmp_path, "profile.toml", toml)
    profile = load_profile(p, git_root=str(tmp_path))
    assert profile["model"]["target"] == "/some/absolute/root/models/base.gguf"


def test_tilde_expands(tmp_path):
    toml = MINIMAL_TOML.replace(
        'target = "models/base.gguf"',
        'target = "~/models/base.gguf"'
    )
    p = _write(tmp_path, "profile.toml", toml)
    profile = load_profile(p, git_root=str(tmp_path))
    home = os.path.expanduser("~")
    assert profile["model"]["target"] == f"{home}/models/base.gguf"


def test_relative_path_resolves_against_git_root(tmp_path):
    toml = MINIMAL_TOML
    p = _write(tmp_path, "profile.toml", toml)
    profile = load_profile(p, git_root=str(tmp_path))
    assert profile["model"]["target"] == str(tmp_path / "models" / "base.gguf")
