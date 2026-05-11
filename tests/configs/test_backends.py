"""Tests for configlib.backends — TDD harness."""
import os
import pytest
from dflash.scripts.configlib.backends import load_backend, build_argv, BackendError


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return p


def _minimal_backend_toml(bin_path, name="dflash", spec_types=None, extra_flags=""):
    spec_types = spec_types or ["none"]
    spec_list = "[" + ", ".join('"' + s + '"' for s in spec_types) + "]"
    flags_for_mtp = ""
    flags_for_dflash = ""
    if "mtp" in spec_types:
        flags_for_mtp = 'spec_model = "--mtp"\nspec_gamma = "--gamma"\n'
    if "dflash" in spec_types:
        flags_for_dflash = 'draft_model = "--draft"\ndraft_max = "--draft-max"\n'
    return (
        'name = "' + name + '"\n'
        '[binary]\n'
        'in_tree = "' + str(bin_path) + '"\n'
        '[supports]\n'
        'spec_types = ' + spec_list + '\n'
        'kv_quants = ["q8_0"]\n'
        '[flags]\n'
        'ctx = "--ctx-size"\n'
        'kv_k = "--kv-k"\n'
        'kv_v = "--kv-v"\n'
        'model = "--model"\n'
        + flags_for_mtp + flags_for_dflash + extra_flags
    )


def test_in_tree_binary_exists_resolves(tmp_path):
    bin_path = tmp_path / "mybin"
    bin_path.touch()
    toml = _minimal_backend_toml(bin_path)
    p = _write(tmp_path, "dflash.toml", toml)
    backend = load_backend(p, git_root=str(tmp_path))
    assert backend["resolved_binary"] == str(bin_path)


def test_in_tree_binary_missing_raises_error(tmp_path):
    toml = _minimal_backend_toml("nonexistent/binary")
    p = _write(tmp_path, "dflash.toml", toml)
    with pytest.raises(BackendError, match="not found"):
        load_backend(p, git_root=str(tmp_path))


def test_env_var_backend_unset_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("LUCEBOX_LLAMA_BIN", raising=False)
    toml = (
        'name = "llama-upstream"\n'
        '[binary]\n'
        'env_var = "LUCEBOX_LLAMA_BIN"\n'
        '[supports]\n'
        'spec_types = ["none"]\n'
        'kv_quants = ["q8_0"]\n'
        '[flags]\n'
        'ctx = "--ctx-size"\n'
        'kv_k = "--kv-cache-type-k"\n'
        'kv_v = "--kv-cache-type-v"\n'
        'model = "--model"\n'
    )
    p = _write(tmp_path, "llama-upstream.toml", toml)
    with pytest.raises(BackendError, match="LUCEBOX_LLAMA_BIN"):
        load_backend(p, git_root=str(tmp_path))


def test_env_var_backend_set_to_existing_resolves(tmp_path, monkeypatch):
    bin_path = tmp_path / "llamabin"
    bin_path.touch()
    monkeypatch.setenv("LUCEBOX_LLAMA_BIN", str(bin_path))
    toml = (
        'name = "llama-upstream"\n'
        '[binary]\n'
        'env_var = "LUCEBOX_LLAMA_BIN"\n'
        '[supports]\n'
        'spec_types = ["none"]\n'
        'kv_quants = ["q8_0"]\n'
        '[flags]\n'
        'ctx = "--ctx-size"\n'
        'kv_k = "--kv-cache-type-k"\n'
        'kv_v = "--kv-cache-type-v"\n'
        'model = "--model"\n'
    )
    p = _write(tmp_path, "llama-upstream.toml", toml)
    backend = load_backend(p, git_root=str(tmp_path))
    assert backend["resolved_binary"] == str(bin_path)


def test_env_var_backend_set_to_nonexistent_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("LUCEBOX_LLAMA_BIN", "/nonexistent/path/llama")
    toml = (
        'name = "llama-upstream"\n'
        '[binary]\n'
        'env_var = "LUCEBOX_LLAMA_BIN"\n'
        '[supports]\n'
        'spec_types = ["none"]\n'
        'kv_quants = ["q8_0"]\n'
        '[flags]\n'
        'ctx = "--ctx-size"\n'
        'kv_k = "--kv-cache-type-k"\n'
        'kv_v = "--kv-cache-type-v"\n'
        'model = "--model"\n'
    )
    p = _write(tmp_path, "llama-upstream.toml", toml)
    with pytest.raises(BackendError, match="does not exist"):
        load_backend(p, git_root=str(tmp_path))


def test_name_mismatch_raises(tmp_path):
    toml = (
        'name = "wrong-name"\n'
        '[binary]\n'
        'in_tree = "somewhere"\n'
        '[supports]\n'
        'spec_types = ["none"]\n'
        'kv_quants = ["q8_0"]\n'
        '[flags]\n'
        'ctx = "--ctx-size"\n'
        'kv_k = "--kv-k"\n'
        'kv_v = "--kv-v"\n'
        'model = "--model"\n'
    )
    p = _write(tmp_path, "dflash.toml", toml)
    with pytest.raises(BackendError, match="name"):
        load_backend(p, git_root=str(tmp_path))


def test_both_in_tree_and_env_var_raises(tmp_path):
    toml = (
        'name = "dflash"\n'
        '[binary]\n'
        'in_tree = "somewhere"\n'
        'env_var = "SOME_VAR"\n'
        '[supports]\n'
        'spec_types = ["none"]\n'
        'kv_quants = ["q8_0"]\n'
        '[flags]\n'
        'ctx = "--ctx-size"\n'
        'kv_k = "--kv-k"\n'
        'kv_v = "--kv-v"\n'
        'model = "--model"\n'
    )
    p = _write(tmp_path, "dflash.toml", toml)
    with pytest.raises(BackendError, match="[Mm]utually exclusive"):
        load_backend(p, git_root=str(tmp_path))


def test_neither_in_tree_nor_env_var_raises(tmp_path):
    toml = (
        'name = "dflash"\n'
        '[binary]\n'
        '[supports]\n'
        'spec_types = ["none"]\n'
        'kv_quants = ["q8_0"]\n'
        '[flags]\n'
        'ctx = "--ctx-size"\n'
        'kv_k = "--kv-k"\n'
        'kv_v = "--kv-v"\n'
        'model = "--model"\n'
    )
    p = _write(tmp_path, "dflash.toml", toml)
    with pytest.raises(BackendError):
        load_backend(p, git_root=str(tmp_path))


def test_missing_flags_for_spec_types_raises(tmp_path):
    toml = (
        'name = "dflash"\n'
        '[binary]\n'
        'in_tree = "somewhere"\n'
        '[supports]\n'
        'spec_types = ["none", "mtp"]\n'
        'kv_quants = ["q8_0"]\n'
        '[flags]\n'
        'ctx = "--ctx-size"\n'
        'kv_k = "--kv-k"\n'
        'kv_v = "--kv-v"\n'
        'model = "--model"\n'
    )
    p = _write(tmp_path, "dflash.toml", toml)
    with pytest.raises(BackendError, match="[Ff]lag"):
        load_backend(p, git_root=str(tmp_path))


def test_build_argv_includes_ctx_kv_flags(tmp_path):
    bin_path = tmp_path / "mybin"
    bin_path.touch()
    toml = _minimal_backend_toml(bin_path)
    p = _write(tmp_path, "dflash.toml", toml)
    backend = load_backend(p, git_root=str(tmp_path))
    profile = {
        "model": {"target": "/model.gguf"},
        "runtime": {"ctx": 4096, "kv_k": "q8_0", "kv_v": "q8_0", "spec": {"method": "none"}},
    }
    argv = build_argv(backend, profile)
    assert "--ctx-size" in argv
    assert "4096" in argv
    assert "--kv-k" in argv
    assert "q8_0" in argv


def test_build_argv_boolean_flag_only_when_true(tmp_path):
    bin_path = tmp_path / "mybin"
    bin_path.touch()
    extra = 'pflash = "--pflash"\nignore_eos = "--ignore-eos"\n'
    toml = _minimal_backend_toml(bin_path, extra_flags=extra)
    p = _write(tmp_path, "dflash.toml", toml)
    backend = load_backend(p, git_root=str(tmp_path))

    profile_false = {
        "model": {"target": "/model.gguf"},
        "runtime": {"ctx": 4096, "kv_k": "q8_0", "kv_v": "q8_0",
                    "pflash": False, "ignore_eos": False, "spec": {"method": "none"}},
    }
    argv = build_argv(backend, profile_false)
    assert "--pflash" not in argv
    assert "--ignore-eos" not in argv

    profile_true = {
        "model": {"target": "/model.gguf"},
        "runtime": {"ctx": 4096, "kv_k": "q8_0", "kv_v": "q8_0",
                    "pflash": True, "ignore_eos": True, "spec": {"method": "none"}},
    }
    argv = build_argv(backend, profile_true)
    assert "--pflash" in argv
    assert "--ignore-eos" in argv


def test_build_argv_mtp_method_adds_mtp_and_gamma(tmp_path):
    bin_path = tmp_path / "mybin"
    bin_path.touch()
    toml = _minimal_backend_toml(bin_path, spec_types=["none", "mtp"])
    p = _write(tmp_path, "dflash.toml", toml)
    backend = load_backend(p, git_root=str(tmp_path))
    profile = {
        "model": {"target": "/model.gguf", "mtp_assistant": "/assistant.gguf"},
        "runtime": {"ctx": 4096, "kv_k": "q8_0", "kv_v": "q8_0",
                    "spec": {"method": "mtp", "gamma": 2}},
    }
    argv = build_argv(backend, profile)
    assert "--mtp" in argv
    assert "/assistant.gguf" in argv
    assert "--gamma" in argv
    assert "2" in argv


def test_build_argv_dflash_method_adds_draft_and_draft_max(tmp_path):
    bin_path = tmp_path / "mybin"
    bin_path.touch()
    toml = _minimal_backend_toml(bin_path, spec_types=["none", "dflash"])
    p = _write(tmp_path, "dflash.toml", toml)
    backend = load_backend(p, git_root=str(tmp_path))
    profile = {
        "model": {"target": "/model.gguf", "dflash_draft": "/draft.gguf"},
        "runtime": {"ctx": 4096, "kv_k": "q8_0", "kv_v": "q8_0",
                    "spec": {"method": "dflash", "draft_max": 4}},
    }
    argv = build_argv(backend, profile)
    assert "--draft" in argv
    assert "/draft.gguf" in argv
    assert "--draft-max" in argv
    assert "4" in argv
