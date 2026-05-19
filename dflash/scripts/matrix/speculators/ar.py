"""AR (autoregressive) speculator — gamma=0 path via test_dflash.

Runs test_dflash with --gamma 0 and --draft-source chain.  This is the
baseline; no draft model is used.
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


class ARSpeculator(Speculator):
    """Pure autoregressive baseline (gamma=0).

    Requires:
        GGUF_PATH  — path to the target model GGUF (env DFLASH_TARGET or explicit).
        DFLASH_BIN — path to test_dflash binary.
        BENCH_KV_TYPE — KV quant (default q8_0).
    """

    def __init__(
        self,
        gguf_path: str | None = None,
        test_dflash: str | None = None,
        kv_type: str | None = None,
        sleep_between_runs: float = 15.0,
    ) -> None:
        # AR baseline target preference: explicit > DFLASH_TARGET env >
        # MTP_GGUF env (the MTP-fused GGUF works as bare backbone since
        # gguf_target_loader subtracts nextn_predict_layers) > fallback path.
        self.gguf_path = gguf_path or os.environ.get(
            "DFLASH_TARGET",
            os.environ.get(
                "MTP_GGUF",
                str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
            ),
        )
        self.test_dflash = test_dflash or os.environ.get(
            "DFLASH_BIN",
            str(ROOT / "build" / f"test_dflash{BIN_SUFFIX}"),
        )
        self.kv_type = kv_type or os.environ.get("BENCH_KV_TYPE", "q8_0")
        self.sleep_between_runs = sleep_between_runs
        self.name = "ar"

    def run(
        self,
        prompt_path: str,
        prompt_id: int,
        n_prompt: int,
        n_gen: int,
        max_ctx: int,
    ) -> SpeculatorResult:
        cmd = [
            self.test_dflash, self.gguf_path,
            "--mtp-gguf", self.gguf_path,   # same file; gamma=0 ignores MTP head
            "--prompt-bin", str(prompt_path),
            "--n-gen", str(n_gen),
            "--gamma", "0",
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
                f"AR (test_dflash gamma=0) exited {r.returncode} "
                f"(prompt_id={prompt_id}):\n{(r.stderr or r.stdout)[-2000:]}"
            )
        data = parse_result_line(r.stdout)
        proposed = data.get("proposed", 0)
        return SpeculatorResult(
            tok_s=data["tok_s"],
            accept_rate=None,         # AR has no speculative acceptance
            al=data.get("al"),
            n_tokens_generated=data["tokens"],
            decode_s=data["decode_s"],
            prefill_s=data["prefill_s"],
            raw_json=data,
        )

    def config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "gguf_path": self.gguf_path,
            "kv_type": self.kv_type,
            "gamma": 0,
        }
