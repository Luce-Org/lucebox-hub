"""MTP chain speculator — test_dflash --draft-source chain --gamma D.

Drives the MtpChainRunner path.  D=3 is the default for mtp_d3.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

from matrix.speculator import Speculator, SpeculatorResult, parse_result_line

ROOT = Path(__file__).resolve().parent.parent.parent.parent   # dflash/
BIN_SUFFIX = ".exe" if os.name == "nt" else ""


class MTPSpeculator(Speculator):
    """MTP chain speculator (IMtpModule-backed, --draft-source chain).

    Parameters
    ----------
    chain_depth:
        Number of MTP heads to chain (--gamma).  Default 3 → "mtp_d3".
    gguf_path:
        Path to MTP GGUF (target + head fused). Falls back to env MTP_GGUF.
    kv_type:
        KV cache quantisation (default q8_0).
    sleep_between_runs:
        WSL2 libggml-cuda.so teardown mitigation sleep in seconds.
    """

    def __init__(
        self,
        chain_depth: int = 3,
        gguf_path: str | None = None,
        kv_type: str | None = None,
        sleep_between_runs: float = 15.0,
    ) -> None:
        self.chain_depth = chain_depth
        self.gguf_path = gguf_path or os.environ.get("MTP_GGUF", "")
        self.test_dflash = os.environ.get(
            "DFLASH_BIN",
            str(ROOT / "build" / f"test_dflash{BIN_SUFFIX}"),
        )
        self.kv_type = kv_type or os.environ.get("BENCH_KV_TYPE", "q8_0")
        self.sleep_between_runs = sleep_between_runs
        self.name = f"mtp_d{chain_depth}"

    def run(
        self,
        prompt_path: str,
        prompt_id: int,
        n_prompt: int,
        n_gen: int,
        max_ctx: int,
    ) -> SpeculatorResult:
        if not self.gguf_path:
            raise RuntimeError(
                "MTPSpeculator requires MTP_GGUF env or explicit gguf_path"
            )
        cmd = [
            self.test_dflash, self.gguf_path,
            "--mtp-gguf", self.gguf_path,
            "--prompt-bin", str(prompt_path),
            "--n-gen", str(n_gen),
            "--gamma", str(self.chain_depth),
            "--prompt-id", str(prompt_id),
            "--max-ctx", str(max_ctx),
            "-ctk", self.kv_type, "-ctv", self.kv_type,
            "--draft-source", "chain",
        ]
        time.sleep(self.sleep_between_runs)
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,
            env=self._make_env(),
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"MTP chain (D={self.chain_depth}) exited {r.returncode} "
                f"(prompt_id={prompt_id}):\n{(r.stderr or r.stdout)[-2000:]}"
            )
        data = parse_result_line(r.stdout)
        proposed = data.get("proposed", 0)
        accepted = data.get("accepted", 0)
        accept_rate = (accepted / proposed) if proposed > 0 else None
        return SpeculatorResult(
            tok_s=data["tok_s"],
            accept_rate=accept_rate,
            al=data.get("al"),
            n_tokens_generated=data["tokens"],
            decode_s=data["decode_s"],
            prefill_s=data["prefill_s"],
            raw_json=data,
        )

    def config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "chain_depth": self.chain_depth,
            "gguf_path": self.gguf_path,
            "kv_type": self.kv_type,
            "draft_source": "chain",
        }
