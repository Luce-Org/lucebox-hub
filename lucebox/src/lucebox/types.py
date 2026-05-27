"""Shared dataclasses passed between modules.

HostFacts is populated from the LUCEBOX_HOST_* env vars set by lucebox.sh.
Config is what we serialize to/from .lucebox/config.toml. Both are frozen so
mistakes (e.g. mutating a config after autotune wrote it) fail loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Variant = str
CtkStatus = Literal["runtime", "cdi", "installed-unwired", "none"]
GpuVendor = Literal["nvidia", "amd", "none"]
AutotuneSource = Literal["heuristic", "benchmark"]


@dataclass(frozen=True, slots=True)
class HostFacts:
    """Probed once by lucebox.sh, passed in via env vars. Single source of
    truth on the Python side — we never reprobe (we can't see host /proc)."""

    nproc: int = 0
    ram_gb: int = 0
    gpu_vendor: GpuVendor = "none"
    gpu_name: str = ""
    gpu_count: int = 0
    vram_gb: int = 0
    gpu_sm: str = ""  # e.g. "120" — matches docker-bake arch lists
    driver_version: str = ""  # e.g. "595.71.05"
    driver_major: int = 0
    has_systemd: bool = False
    is_wsl: bool = False
    has_docker: bool = False
    docker_version: str = ""
    ctk: CtkStatus = "none"


@dataclass(frozen=True, slots=True)
class DflashRuntime:
    """The DFLASH_* knobs as typed values. Serialized under [dflash] in TOML
    and emitted as -e DFLASH_FOO=bar args to docker run."""

    budget: int = 22
    max_ctx: int = 16384
    lazy: bool = False
    prefix_cache_slots: int = 0
    prefill_cache_slots: int = 0
    cache_type_k: str = ""
    cache_type_v: str = ""
    prefill_mode: Literal["off", "auto", "always"] = "off"
    prefill_keep_ratio: float = 0.05
    prefill_threshold: int = 32000
    prefill_drafter: str = ""
    # Phase-1 (thinking) cap when a request opts into thinking. Default mirrors
    # antirez/ds4 ds4_eval.c: think_max_tokens = max_tokens - hard_limit_reply
    # budget = 16000 - 512 = 15488. The server's own hardcoded default is 10000.
    think_max: int = 15488


@dataclass(frozen=True, slots=True)
class AutotuneMeta:
    source: AutotuneSource = "heuristic"
    timestamp: str = ""  # ISO-8601


@dataclass(frozen=True, slots=True)
class BenchmarkMeta:
    """Filled in by the optimizer; absent until the user runs `benchmark`."""

    ran_at: str = ""
    profile: str = ""  # e.g. "he-decode"
    winner_budget: int | None = None
    winner_max_ctx: int | None = None
    winner_lazy: bool | None = None
    winner_prefix_cache_slots: int | None = None
    winner_prefill_cache_slots: int | None = None
    winner_cache_type_k: str = ""
    winner_cache_type_v: str = ""
    winner_prefill_mode: str = ""
    mean_tps: float | None = None
    report_path: str = ""  # relative to config dir


@dataclass(frozen=True, slots=True)
class Config:
    """The whole config.toml, materialized."""

    variant: Variant = "cuda12"
    image: str = "ghcr.io/luce-org/lucebox-hub"
    container_name: str = "lucebox"
    port: int = 8080
    models_dir: Path = Path.home() / "models"
    dflash: DflashRuntime = field(default_factory=DflashRuntime)
    host: HostFacts = field(default_factory=HostFacts)
    autotune: AutotuneMeta = field(default_factory=AutotuneMeta)
    benchmark: BenchmarkMeta | None = None
