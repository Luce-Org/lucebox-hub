"""Model download orchestration.

Runs *inside* the orchestrator container — uses `uvx --from
'huggingface_hub[cli]' hf download` so we don't pollute the lucebox venv
with HF deps. Writes into the host-bind-mounted models dir (config.models_dir).

Default preset: unsloth/Qwen3.6-27B-GGUF Q4_K_M target + Q4_K_M GGUF
DFlash draft. Single preset for v1; future versions will accept --model.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from lucebox.types import Config


@dataclass(frozen=True, slots=True)
class ModelPreset:
    name: str
    target_repo: str
    target_file: str
    draft_repo: str
    draft_file: str
    approx_total_gb: int


DEFAULT_PRESET = ModelPreset(
    name="qwen3.6-27b",
    target_repo="unsloth/Qwen3.6-27B-GGUF",
    target_file="Qwen3.6-27B-Q4_K_M.gguf",
    draft_repo="spiritbuun/Qwen3.6-27B-DFlash-GGUF",
    draft_file="dflash-draft-3.6-q4_k_m.gguf",
    approx_total_gb=17,
)

MIN_TARGET_BYTES = 1024 * 1024 * 1024
MIN_DRAFT_BYTES = 1024 * 1024


def _hf_download(args: list[str]) -> int:
    """Run `uvx --from 'huggingface_hub[cli]' hf download …`, streaming output."""
    cmd = ["uvx", "--from", "huggingface_hub[cli]", "hf", "download", *args]
    return subprocess.call(cmd)


def download_preset(cfg: Config, preset: ModelPreset = DEFAULT_PRESET) -> int:
    """Fetch the target GGUF + DFlash draft into cfg.models_dir.

    Returns 0 on success, non-zero on failure (skips partial cleanup — HF's
    resume support makes a re-run cheap).
    """
    if shutil.which("uvx") is None:
        raise RuntimeError("uvx not available in this container — expected uv install")

    models = cfg.models_dir
    models.mkdir(parents=True, exist_ok=True)
    draft = models / "draft"
    draft.mkdir(exist_ok=True)

    target_path = models / preset.target_file
    if target_path.exists() and target_path.stat().st_size > MIN_TARGET_BYTES:
        # Already downloaded (>1 GB suggests it's not a partial). Skip.
        pass
    else:
        rc = _hf_download(
            [
                preset.target_repo,
                preset.target_file,
                "--local-dir",
                str(models),
            ]
        )
        if rc != 0:
            return rc

    draft_path = draft / preset.draft_file
    if draft_path.exists() and draft_path.stat().st_size > MIN_DRAFT_BYTES:
        pass
    else:
        rc = _hf_download(
            [
                preset.draft_repo,
                preset.draft_file,
                "--local-dir",
                str(draft),
            ]
        )
        if rc != 0:
            return rc

    return 0


def status(cfg: Config, preset: ModelPreset = DEFAULT_PRESET) -> dict[str, bool]:
    """Quick presence check — what's already on disk?"""
    target = cfg.models_dir / preset.target_file
    draft = cfg.models_dir / "draft" / preset.draft_file
    return {
        "target_present": target.exists() and target.stat().st_size > MIN_TARGET_BYTES,
        "draft_present": draft.exists() and draft.stat().st_size > MIN_DRAFT_BYTES,
    }
