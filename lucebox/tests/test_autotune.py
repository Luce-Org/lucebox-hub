from lucebox.autotune import runtime_from_host
from lucebox.types import HostFacts


def test_wsl_24gb_defaults_leave_cuda_headroom() -> None:
    runtime = runtime_from_host(HostFacts(vram_gb=24, is_wsl=True))

    assert runtime.budget == 16
    assert runtime.max_ctx == 65536
    assert runtime.lazy is True
    assert runtime.prefix_cache_slots == 0


def test_native_24gb_caps_context_below_vmm_failure_boundary() -> None:
    runtime = runtime_from_host(HostFacts(vram_gb=24, is_wsl=False))

    assert runtime.budget == 22
    assert runtime.max_ctx == 98304
    assert runtime.lazy is True
    assert runtime.prefix_cache_slots == 0
