from __future__ import annotations

"""Server-side placement resolution before daemon launch."""

from dataclasses import dataclass
from typing import MutableMapping

from .backend_device import apply_backend_visible_devices
from .test_dflash_args import TestDflashLaunchArgs


@dataclass(frozen=True)
class ServerPlacement:
    env_updates: dict[str, str]
    daemon_args: list[str]
    prefix_cache_slots: int
    prefill_cache_slots: int
    target_backend: str
    draft_backend: str
    target_visible_devices: str | None
    draft_visible_devices: str | None
    target_gpu: int | None
    draft_gpu: int | None
    target_gpus: str | None
    target_layer_split: str | None
    draft_ipc_bin: str | None
    draft_ipc_gpu: int | None
    draft_ipc_work_dir: str | None
    draft_ipc_ring_cap: int | None
    draft_feature_mirror: bool
    peer_access: bool
    cache_slots_disabled: bool = False

    @property
    def mixed_backend(self) -> bool:
        return self.target_backend != self.draft_backend

    def apply_env(self, env: MutableMapping[str, str]) -> None:
        env.update(self.env_updates)

    def log_lines(self) -> list[str]:
        lines = [
            f"  placement = {'target-gpus' if self.target_gpus else 'single-target'}",
            f"    target_backend = {self.target_backend}",
            f"    draft_backend  = {self.draft_backend}",
        ]
        if self.target_visible_devices:
            lines.append(f"    target_visible_devices = {self.target_visible_devices}")
        if self.draft_visible_devices:
            lines.append(f"    draft_visible_devices  = {self.draft_visible_devices}")
        if self.target_gpu is not None:
            lines.append(f"    target_gpu = {self.target_gpu}")
        if self.draft_gpu is not None:
            lines.append(f"    draft_gpu  = {self.draft_gpu}")
        if self.target_gpus:
            lines.append(f"    target_gpus = {self.target_gpus}")
            lines.append(f"    layer_split = {self.target_layer_split or '<default>'}")
        if self.draft_ipc_bin:
            lines.append(f"    draft_ipc_bin = {self.draft_ipc_bin}")
            if self.draft_ipc_gpu is not None:
                lines.append(f"    draft_ipc_gpu = {self.draft_ipc_gpu}")
            if self.draft_ipc_ring_cap is not None:
                lines.append(f"    draft_ipc_ring_cap = {self.draft_ipc_ring_cap}")
        if self.draft_feature_mirror:
            lines.append("    draft_feature_mirror = on")
        if self.peer_access:
            lines.append("    peer_access = on")
        return lines


def _add_backend_visible_env(env_updates: dict[str, str],
                             backend: str,
                             visible_devices: str | None) -> None:
    if not visible_devices:
        return
    updates = apply_backend_visible_devices(
        backend,
        visible_devices=visible_devices,
        base_env={},
    )
    conflicts = [
        key for key, value in updates.items()
        if key in env_updates and env_updates[key] != value
    ]
    if conflicts:
        names = ", ".join(conflicts)
        raise ValueError(
            f"conflicting visible-device placement for {backend}: {names}")
    env_updates.update(updates)


def resolve_server_placement(args) -> ServerPlacement:
    env_updates: dict[str, str] = {}
    target_backend = args.target_backend
    draft_backend = args.draft_backend
    mixed_backend = target_backend != draft_backend

    if (target_backend == draft_backend and
            args.target_visible_devices and args.draft_visible_devices and
            args.target_visible_devices != args.draft_visible_devices):
        raise ValueError(
            "same-backend target/draft placement cannot use different visible "
            "device lists in one server process; use --target-gpu/--draft-gpu "
            "within the shared visible-device list")

    _add_backend_visible_env(
        env_updates, target_backend, args.target_visible_devices)
    _add_backend_visible_env(
        env_updates, draft_backend, args.draft_visible_devices)

    if args.target_gpu is not None:
        env_updates["DFLASH_TARGET_GPU"] = str(args.target_gpu)
    if args.draft_gpu is not None:
        env_updates["DFLASH_DRAFT_GPU"] = str(args.draft_gpu)

    if mixed_backend and not args.draft_ipc_bin:
        raise ValueError(
            "mixed-backend draft/target placement requires --draft-ipc-bin "
            "pointing to a test_dflash binary built for the draft backend")
    if not args.draft_ipc_bin and (
            args.draft_ipc_gpu is not None or
            args.draft_ipc_work_dir is not None or
            args.draft_ipc_ring_cap is not None):
        raise ValueError(
            "--draft-ipc-gpu, --draft-ipc-work-dir, and "
            "--draft-ipc-ring-cap require --draft-ipc-bin")
    if args.draft_ipc_bin and not args.target_gpus:
        raise ValueError(
            "--draft-ipc-bin requires --target-gpus because test_dflash "
            "draft IPC is implemented for the target-split DFlash daemon path")

    daemon_cfg = TestDflashLaunchArgs(
        draft_feature_mirror=args.draft_feature_mirror,
        peer_access=args.peer_access,
        draft_ipc_bin=str(args.draft_ipc_bin) if args.draft_ipc_bin else None,
        draft_ipc_gpu=args.draft_ipc_gpu,
        draft_ipc_work_dir=str(args.draft_ipc_work_dir) if args.draft_ipc_work_dir else None,
        draft_ipc_ring_cap=args.draft_ipc_ring_cap,
    )

    prefix_cache_slots = args.prefix_cache_slots
    prefill_cache_slots = args.prefill_cache_slots
    cache_slots_disabled = False

    if args.target_gpus:
        daemon_cfg = TestDflashLaunchArgs(
            draft_feature_mirror=args.draft_feature_mirror,
            peer_access=args.peer_access,
            target_gpus=args.target_gpus,
            target_layer_split=args.target_layer_split,
            target_split_load_draft=True,
            target_split_dflash=True,
            draft_ipc_bin=str(args.draft_ipc_bin) if args.draft_ipc_bin else None,
            draft_ipc_gpu=args.draft_ipc_gpu,
            draft_ipc_work_dir=str(args.draft_ipc_work_dir) if args.draft_ipc_work_dir else None,
            draft_ipc_ring_cap=args.draft_ipc_ring_cap,
        )
        cache_slots_disabled = prefix_cache_slots > 0 or prefill_cache_slots > 0
        if cache_slots_disabled:
            prefix_cache_slots = 0
            prefill_cache_slots = 0

    return ServerPlacement(
        env_updates=env_updates,
        daemon_args=daemon_cfg.to_cli_args(),
        prefix_cache_slots=prefix_cache_slots,
        prefill_cache_slots=prefill_cache_slots,
        target_backend=target_backend,
        draft_backend=draft_backend,
        target_visible_devices=args.target_visible_devices,
        draft_visible_devices=args.draft_visible_devices,
        target_gpu=args.target_gpu,
        draft_gpu=args.draft_gpu,
        target_gpus=args.target_gpus,
        target_layer_split=args.target_layer_split,
        draft_ipc_bin=str(args.draft_ipc_bin) if args.draft_ipc_bin else None,
        draft_ipc_gpu=args.draft_ipc_gpu,
        draft_ipc_work_dir=str(args.draft_ipc_work_dir) if args.draft_ipc_work_dir else None,
        draft_ipc_ring_cap=args.draft_ipc_ring_cap,
        draft_feature_mirror=args.draft_feature_mirror,
        peer_access=args.peer_access,
        cache_slots_disabled=cache_slots_disabled,
    )
