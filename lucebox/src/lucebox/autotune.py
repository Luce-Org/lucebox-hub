"""Heuristic autotune: VRAM tier → DflashRuntime defaults.

Kept in one place here so the benchmark optimizer and the on-the-fly
`configure` both share the same rules.

The benchmark optimizer overrides specific fields (today: `budget`) via
`merge_benchmark_winner`. Heuristic tiers fill everything; benchmark refines.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, cast

from lucebox.types import DflashRuntime, HostFacts


def runtime_from_host(host: HostFacts) -> DflashRuntime:
    """Pick a conservative DflashRuntime that 'should work' on this VRAM tier.

    Tiers (NVIDIA, target = Qwen3.6-27B Q4_K_M ~16 GB + Q4_K_M DFlash
    draft ~1 GB):
        <12 GB  — too small for 27B; pick min ctx and lazy draft as a floor
                  so a fallback start at least gets an error from the daemon
                  rather than a silent OOM.
        12-21   — fits but tight; cap ctx, keep draft lazy.
        22-31   — 24 GB-class consumer flagships (3090/4090/5090/5090-Laptop).
                  Cap at 96 K by default; 112 K can start, but long prompts
                  have reproduced CUDA VMM allocation failures on 24 GB cards.
                  WSL needs more CUDA/VMM headroom, so it starts lower and lets
                  the stress optimizer prove higher settings before persisting.
        32-47   — RTX 6000 Ada / A100 40 GB. Full 128 K.
        ≥48     — A100 80 GB / H100 / RTX 6000 Pro. Full 128 K.

    Prefix cache remains an explicit benchmark tunable, but the automatic
    baseline keeps it off because tool prompts currently exercise a daemon
    snapshot path that is not reliable with prefix slots enabled.
    """
    if host.vram_gb <= 0:
        return DflashRuntime()  # no VRAM signal — stick with class defaults

    if host.vram_gb < 12:
        return DflashRuntime(lazy=True, max_ctx=4096)
    if host.vram_gb < 22:
        return DflashRuntime(lazy=True, max_ctx=32768)
    if host.vram_gb < 32:
        if host.is_wsl:
            return DflashRuntime(budget=16, lazy=True, max_ctx=65536)
        return DflashRuntime(lazy=True, max_ctx=98304)
    if host.vram_gb < 48:
        return DflashRuntime(max_ctx=131072)
    return DflashRuntime(max_ctx=131072)


def merge_benchmark_winner(
    base: DflashRuntime,
    *,
    budget: int,
    max_ctx: int | None = None,
    lazy: bool | None = None,
    prefix_cache_slots: int | None = None,
    prefill_cache_slots: int | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    prefill_mode: str | None = None,
    prefill_keep_ratio: float | None = None,
    prefill_threshold: int | None = None,
    prefill_drafter: str | None = None,
) -> DflashRuntime:
    """Apply benchmark-discovered overrides on top of the heuristic baseline.

    The optimizer can refine both the speculative budget and the verified
    context length. Future sweeps can extend this with prefix/pFlash knobs.
    """
    return replace(
        base,
        budget=budget,
        max_ctx=base.max_ctx if max_ctx is None else max_ctx,
        lazy=base.lazy if lazy is None else lazy,
        prefix_cache_slots=(
            base.prefix_cache_slots if prefix_cache_slots is None else prefix_cache_slots
        ),
        prefill_cache_slots=(
            base.prefill_cache_slots if prefill_cache_slots is None else prefill_cache_slots
        ),
        cache_type_k=base.cache_type_k if cache_type_k is None else cache_type_k,
        cache_type_v=base.cache_type_v if cache_type_v is None else cache_type_v,
        prefill_mode=cast(
            Literal["off", "auto", "always"],
            base.prefill_mode if prefill_mode is None else prefill_mode,
        ),
        prefill_keep_ratio=(
            base.prefill_keep_ratio if prefill_keep_ratio is None else prefill_keep_ratio
        ),
        prefill_threshold=(
            base.prefill_threshold if prefill_threshold is None else prefill_threshold
        ),
        prefill_drafter=base.prefill_drafter if prefill_drafter is None else prefill_drafter,
    )
