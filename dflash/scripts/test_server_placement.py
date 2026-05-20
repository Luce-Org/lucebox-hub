from pathlib import Path
from types import SimpleNamespace

import pytest

from placement.server_resolver import resolve_server_placement


def _args(**overrides):
    base = dict(
        target_backend="cuda",
        draft_backend="cuda",
        target_visible_devices=None,
        draft_visible_devices=None,
        target_gpu=None,
        draft_gpu=None,
        target_gpus=None,
        target_layer_split=None,
        draft_feature_mirror=False,
        peer_access=False,
        draft_ipc_bin=None,
        draft_ipc_gpu=None,
        draft_ipc_work_dir=None,
        draft_ipc_ring_cap=None,
        prefix_cache_slots=1,
        prefill_cache_slots=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_same_backend_visible_devices_share_one_process_env():
    placement = resolve_server_placement(_args(
        target_visible_devices="0,1",
        target_gpu=0,
        draft_gpu=1,
    ))

    assert placement.env_updates["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert placement.env_updates["DFLASH_TARGET_GPU"] == "0"
    assert placement.env_updates["DFLASH_DRAFT_GPU"] == "1"
    assert placement.mixed_backend is False


def test_same_backend_rejects_conflicting_visible_devices():
    with pytest.raises(ValueError, match="same-backend target/draft placement"):
        resolve_server_placement(_args(
            target_visible_devices="0",
            draft_visible_devices="1",
        ))


def test_mixed_backend_requires_draft_ipc_binary():
    with pytest.raises(ValueError, match="requires --draft-ipc-bin"):
        resolve_server_placement(_args(
            target_backend="cuda",
            draft_backend="hip",
            target_visible_devices="0,1",
            draft_visible_devices="0",
        ))


def test_mixed_backend_exports_both_backend_visible_envs_and_ipc_args():
    placement = resolve_server_placement(_args(
        target_backend="cuda",
        draft_backend="hip",
        target_visible_devices="0,1",
        draft_visible_devices="0",
        target_gpus="0,1",
        target_layer_split="1,1",
        draft_ipc_bin=Path("/opt/lucebox/hip/test_dflash"),
        draft_ipc_gpu=0,
        draft_ipc_work_dir=Path("/tmp/dflash-ipc"),
        draft_ipc_ring_cap=8192,
    ))

    assert placement.mixed_backend is True
    assert placement.env_updates["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert placement.env_updates["HIP_VISIBLE_DEVICES"] == "0"
    assert placement.env_updates["ROCR_VISIBLE_DEVICES"] == "0"
    assert "--target-gpus=0,1" in placement.daemon_args
    assert "--target-layer-split=1,1" in placement.daemon_args
    assert "--target-split-load-draft" in placement.daemon_args
    assert "--target-split-dflash" in placement.daemon_args
    assert "--draft-ipc-bin=/opt/lucebox/hip/test_dflash" in placement.daemon_args
    assert "--draft-ipc-gpu=0" in placement.daemon_args
    assert "--draft-ipc-work-dir=/tmp/dflash-ipc" in placement.daemon_args
    assert "--draft-ipc-ring-cap=8192" in placement.daemon_args
    assert placement.prefix_cache_slots == 0
    assert placement.cache_slots_disabled is True
