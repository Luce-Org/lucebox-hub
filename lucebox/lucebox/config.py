"""TOML serialization for .lucebox/config.toml.

Single source of truth for the runtime configuration. Both `configure` and
`benchmark` write here; `serve` reads to emit the docker-run argv.

We migrate from the old shell-sourceable `.env` format if we see one on load
— write back as TOML in the same call so the next round skips the migration.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import tomli_w

from lucebox.types import (
    AutotuneMeta,
    AutotuneSource,
    BenchmarkMeta,
    Config,
    DflashRuntime,
    HostFacts,
    Variant,
)


def default_config_path() -> Path:
    """Where .lucebox/config.toml lives.

    Convention: under $LUCEBOX_HOME if set, otherwise $HOME/.lucebox. Lives in
    the bind-mounted host home dir so the config survives container teardown
    and is editable from the host.
    """
    base = os.environ.get("LUCEBOX_HOME")
    if base:
        return Path(base) / "config.toml"
    return Path.home() / ".lucebox" / "config.toml"


# ── load ───────────────────────────────────────────────────────────────────

def load(path: Path | None = None) -> Config | None:
    """Load config.toml, or return None if missing.

    If a legacy `.env` sits next to it (or in place of it), migrate that
    first and write back as TOML.
    """
    path = path or default_config_path()
    if path.exists():
        return _load_toml(path)

    legacy = path.with_suffix(".env")
    if legacy.exists():
        cfg = _load_legacy_env(legacy)
        save(cfg, path)
        return cfg

    return None


def _load_toml(path: Path) -> Config:
    raw = tomllib.loads(path.read_text())
    return _from_dict(raw)


_LEGACY_KEY_MAP: dict[str, tuple[str, str, Callable[[str], Any]]] = {
    "DFLASH_BUDGET":             ("dflash", "budget", int),
    "DFLASH_MAX_CTX":            ("dflash", "max_ctx", int),
    "DFLASH_LAZY":               ("dflash", "lazy", lambda v: v in ("1", "true", "yes")),
    "DFLASH_PREFIX_CACHE_SLOTS": ("dflash", "prefix_cache_slots", int),
    "DFLASH_PORT":               ("runtime", "port", int),
    "LUCEBOX_VARIANT":           ("image", "variant", str),
    "LUCEBOX_IMAGE":             ("image", "registry", str),
    "LUCEBOX_MODELS":            ("paths", "models", str),
}


def _load_legacy_env(path: Path) -> Config:
    """Best-effort migration from the bash-era .lucebox/config.env."""
    raw: dict[str, Any] = {}
    line_re = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)$")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = line_re.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip().strip('"').strip("'")
        if key not in _LEGACY_KEY_MAP:
            continue
        section, field, cast_fn = _LEGACY_KEY_MAP[key]
        try:
            raw.setdefault(section, {})[field] = cast_fn(val)
        except (TypeError, ValueError):
            continue
    return _from_dict(_remap_legacy(raw))


def _remap_legacy(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate the flat legacy layout into the new TOML schema shape."""
    out: dict[str, Any] = {}
    img = raw.get("image", {})
    if "variant" in img:
        out.setdefault("image", {})["variant"] = img["variant"]
    if "registry" in img:
        out.setdefault("image", {})["registry"] = img["registry"]
    if (paths := raw.get("paths", {})).get("models"):
        out.setdefault("paths", {})["models"] = paths["models"]
    if (runtime := raw.get("runtime", {})).get("port"):
        out.setdefault("runtime", {})["port"] = runtime["port"]
    if dflash := raw.get("dflash", {}):
        out["dflash"] = dflash
    out.setdefault("autotune", {"source": "heuristic", "timestamp": _now()})
    return out


def _from_dict(raw: dict[str, Any]) -> Config:
    img = raw.get("image", {})
    variant: Variant = str(img.get("variant", "cuda12"))
    registry = img.get("registry", "ghcr.io/luce-org/lucebox-hub")

    runtime = raw.get("runtime", {})
    port = int(runtime.get("port", 8080))
    container_name = str(runtime.get("container_name", "lucebox"))

    paths = raw.get("paths", {})
    models_dir = Path(paths.get("models", str(Path.home() / "models")))

    df = raw.get("dflash", {})
    dflash = DflashRuntime(
        budget=int(df.get("budget", 22)),
        max_ctx=int(df.get("max_ctx", 16384)),
        lazy=bool(df.get("lazy", False)),
        prefix_cache_slots=int(df.get("prefix_cache_slots", 0)),
        prefill_cache_slots=int(df.get("prefill_cache_slots", 0)),
        cache_type_k=str(df.get("cache_type_k", "")),
        cache_type_v=str(df.get("cache_type_v", "")),
        prefill_mode=df.get("prefill_mode", "off"),
        prefill_keep_ratio=float(df.get("prefill_keep_ratio", 0.05)),
        prefill_threshold=int(df.get("prefill_threshold", 32000)),
        prefill_drafter=str(df.get("prefill_drafter", "")),
        think_max=int(df.get("think_max", 15488)),
    )

    host_raw = raw.get("host", {})
    host = HostFacts(
        nproc=int(host_raw.get("nproc", 0)),
        ram_gb=int(host_raw.get("ram_gb", 0)),
        gpu_vendor=host_raw.get("gpu_vendor", "none"),
        gpu_name=str(host_raw.get("gpu_name", "")),
        gpu_count=int(host_raw.get("gpu_count", 0)),
        vram_gb=int(host_raw.get("vram_gb", 0)),
        gpu_sm=str(host_raw.get("gpu_sm", "")),
        driver_version=str(host_raw.get("driver_version", "")),
        driver_major=int(host_raw.get("driver_major", 0)),
        has_systemd=bool(host_raw.get("has_systemd", False)),
        is_wsl=bool(host_raw.get("is_wsl", False)),
        has_docker=bool(host_raw.get("has_docker", False)),
        docker_version=str(host_raw.get("docker_version", "")),
        ctk=host_raw.get("ctk", "none"),
    )

    at = raw.get("autotune", {})
    source: AutotuneSource = cast(AutotuneSource, at.get("source", "heuristic"))
    autotune = AutotuneMeta(source=source, timestamp=str(at.get("timestamp", "")))

    bench: BenchmarkMeta | None = None
    if bm := raw.get("benchmark"):
        bench = BenchmarkMeta(
            ran_at=str(bm.get("ran_at", "")),
            profile=str(bm.get("profile", "")),
            winner_budget=bm.get("winner_budget"),
            winner_max_ctx=bm.get("winner_max_ctx"),
            winner_lazy=bm.get("winner_lazy"),
            winner_prefix_cache_slots=bm.get("winner_prefix_cache_slots"),
            winner_prefill_cache_slots=bm.get("winner_prefill_cache_slots"),
            winner_cache_type_k=str(bm.get("winner_cache_type_k", "")),
            winner_cache_type_v=str(bm.get("winner_cache_type_v", "")),
            winner_prefill_mode=str(bm.get("winner_prefill_mode", "")),
            mean_tps=bm.get("mean_tps"),
            report_path=str(bm.get("report_path", "")),
        )

    return Config(
        variant=variant,
        image=registry,
        container_name=container_name,
        port=port,
        models_dir=models_dir,
        dflash=dflash,
        host=host,
        autotune=autotune,
        benchmark=bench,
    )


# ── save ───────────────────────────────────────────────────────────────────

def save(cfg: Config, path: Path | None = None) -> Path:
    path = path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_dict(cfg)
    # Write atomically: tomli_w → bytes via a tmp file → rename.
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_bytes(tomli_w.dumps(payload).encode("utf-8"))
    tmp.replace(path)
    return path


def _to_dict(cfg: Config) -> dict[str, Any]:
    out: dict[str, Any] = {
        "image": {
            "variant": cfg.variant,
            "registry": cfg.image,
        },
        "runtime": {
            "port": cfg.port,
            "container_name": cfg.container_name,
        },
        "paths": {
            "models": str(cfg.models_dir),
        },
        "dflash": asdict(cfg.dflash),
        "host": asdict(cfg.host),
        "autotune": asdict(cfg.autotune),
    }
    if cfg.benchmark is not None:
        # asdict drops None fields; tomli-w can't serialize None, so be explicit.
        bm = {k: v for k, v in asdict(cfg.benchmark).items() if v is not None}
        out["benchmark"] = bm
    return out


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
