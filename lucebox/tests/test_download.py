from types import SimpleNamespace

from lucebox.download import DEFAULT_PRESET, status

from lucebox import download


def test_default_preset_uses_quantized_gguf_draft():
    assert DEFAULT_PRESET.draft_repo == "spiritbuun/Qwen3.6-27B-DFlash-GGUF"
    assert DEFAULT_PRESET.draft_file == "dflash-draft-3.6-q4_k_m.gguf"


def test_status_checks_default_draft_gguf(tmp_path):
    cfg = SimpleNamespace(models_dir=tmp_path)
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    target = tmp_path / DEFAULT_PRESET.target_file
    draft = draft_dir / DEFAULT_PRESET.draft_file

    assert status(cfg) == {"target_present": False, "draft_present": False}

    with target.open("wb") as f:
        f.truncate(download.MIN_TARGET_BYTES + 1)
    with draft.open("wb") as f:
        f.truncate(download.MIN_DRAFT_BYTES + 1)

    assert status(cfg) == {"target_present": True, "draft_present": True}


def test_status_rejects_partial_model_files(tmp_path):
    cfg = SimpleNamespace(models_dir=tmp_path)
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    target = tmp_path / DEFAULT_PRESET.target_file
    draft = draft_dir / DEFAULT_PRESET.draft_file
    target.write_bytes(b"partial")
    draft.write_bytes(b"partial")

    assert status(cfg) == {"target_present": False, "draft_present": False}


def test_download_preset_fetches_exact_draft_file(tmp_path, monkeypatch):
    cfg = SimpleNamespace(models_dir=tmp_path)
    calls = []

    monkeypatch.setattr(download.shutil, "which", lambda _: "/usr/bin/uvx")
    monkeypatch.setattr(download, "_hf_download", lambda args: calls.append(args) or 0)

    assert download.download_preset(cfg) == 0

    assert [
        DEFAULT_PRESET.target_repo,
        DEFAULT_PRESET.target_file,
        "--local-dir",
        str(tmp_path),
    ] in calls
    assert [
        DEFAULT_PRESET.draft_repo,
        DEFAULT_PRESET.draft_file,
        "--local-dir",
        str(tmp_path / "draft"),
    ] in calls
