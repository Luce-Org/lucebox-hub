"""DFlash standalone speculator — separate drafter GGUF via test_dflash.

Runs test_dflash with a dedicated draft model (DFLASH_DRAFT env or explicit)
and a DDTree budget.  Budget 22 → "dflash_b22" (matches today's bench_agent.py).
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

import re
from matrix.speculator import Speculator, SpeculatorResult, parse_result_line

# DFlash standalone (test_dflash <target> <draft> ... mode) emits a different
# stdout format than the MTP RESULT_JSON. Mirror bench_agent.py's regexes.
_DFLASH_GEN_RE = re.compile(
    r"\[dflash\] generated (\d+) tokens in ([\d.]+) s\s*->\s*([\d.]+) tok/s"
)
_DFLASH_STEPS_RE = re.compile(
    r"\[dflash\] (\d+) draft steps,\s*accepted=(\d+)/(\d+).*avg commit/step=([\d.]+)"
)
_DFLASH_PREFILL_RE = re.compile(r"prefill[:= ]?\s*([\d.]+)\s*s")


def _parse_dflash_stdout(stdout: str) -> Dict[str, Any]:
    """Parse DFlash standalone (non-MTP) test_dflash stdout."""
    g = _DFLASH_GEN_RE.search(stdout)
    s = _DFLASH_STEPS_RE.search(stdout)
    p = _DFLASH_PREFILL_RE.search(stdout)
    if not g:
        raise RuntimeError(f"DFlash stdout: no [dflash] generated line found\n{stdout[-1500:]}")
    n_tok = int(g.group(1))
    decode_s = float(g.group(2))
    tok_s = float(g.group(3))
    accepted = int(s.group(2)) if s else 0
    proposed = int(s.group(3)) if s else 0
    al = float(s.group(4)) if s else None
    prefill_s = float(p.group(1)) if p else 0.0
    return {
        "tok_s": tok_s, "tokens": n_tok, "decode_s": decode_s, "prefill_s": prefill_s,
        "accepted": accepted, "proposed": proposed, "al": al,
    }

ROOT = Path(__file__).resolve().parent.parent.parent.parent   # dflash/
BIN_SUFFIX = ".exe" if os.name == "nt" else ""

_DEFAULT_DRAFT_SEARCH = [
    ROOT / "models" / "draft" / "dflash-draft-3.6-q8_0.gguf",
    ROOT / "models" / "draft",
]


def _find_draft(root: Path) -> str | None:
    if root.is_file():
        return str(root)
    if root.is_dir():
        for pattern in ("dflash-draft-*.gguf", "*.gguf"):
            matches = sorted(root.rglob(pattern))
            if matches:
                return str(matches[0])
    return None


class DFlashSpeculator(Speculator):
    """DFlash speculator with separate drafter GGUF and DDTree.

    Parameters
    ----------
    budget:
        DDTree node budget.  Default 22 → name "dflash_b22".
    target_gguf:
        Path to target model GGUF. Falls back to env DFLASH_TARGET.
    draft_gguf:
        Path to drafter GGUF. Falls back to env DFLASH_DRAFT then local search.
    kv_type:
        KV cache quantisation (default q8_0).
    sleep_between_runs:
        WSL2 libggml-cuda.so teardown mitigation sleep in seconds.
    """

    def __init__(
        self,
        budget: int = 22,
        target_gguf: str | None = None,
        draft_gguf: str | None = None,
        kv_type: str | None = None,
        sleep_between_runs: float = 15.0,
    ) -> None:
        self.budget = budget
        self.target_gguf = target_gguf or os.environ.get(
            "DFLASH_TARGET",
            str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
        )
        # Resolve draft
        env_draft = os.environ.get("DFLASH_DRAFT", "")
        if draft_gguf:
            self.draft_gguf = draft_gguf
        elif env_draft:
            found = _find_draft(Path(env_draft))
            self.draft_gguf = found or env_draft
        else:
            self.draft_gguf = ""
            for candidate in _DEFAULT_DRAFT_SEARCH:
                found = _find_draft(candidate)
                if found:
                    self.draft_gguf = found
                    break

        self.test_dflash = os.environ.get(
            "DFLASH_BIN",
            str(ROOT / "build" / f"test_dflash{BIN_SUFFIX}"),
        )
        self.kv_type = kv_type or os.environ.get("BENCH_KV_TYPE", "q8_0")
        self.sleep_between_runs = sleep_between_runs
        self.name = f"dflash_b{budget}"

    def run(
        self,
        prompt_path: str,
        prompt_id: int,
        n_prompt: int,
        n_gen: int,
        max_ctx: int,
    ) -> SpeculatorResult:
        if not self.draft_gguf:
            raise RuntimeError(
                "DFlashSpeculator requires a draft GGUF. "
                "Set DFLASH_DRAFT env or pass draft_gguf explicitly."
            )
        # DFlash standalone uses POSITIONAL args (matches bench_agent.py):
        #   test_dflash <target> <draft> <prompt_bin> <n_gen> <out_bin> --flags
        # Flag-style (--prompt-bin) is only for the MTP harness path.
        out_bin = f"/tmp/dflash_matrix_out_{prompt_id}.bin"
        cmd = [
            self.test_dflash, self.target_gguf, self.draft_gguf,
            str(prompt_path), str(n_gen), out_bin,
            "--max-ctx", str(max_ctx),
            "-ctk", self.kv_type, "-ctv", self.kv_type,
            "--fast-rollback", "--ddtree",
            f"--ddtree-budget={self.budget}",
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
                f"DFlash (budget={self.budget}) exited {r.returncode} "
                f"(prompt_id={prompt_id}):\n{(r.stderr or r.stdout)[-2000:]}"
            )
        data = _parse_dflash_stdout(r.stdout)
        proposed = data.get("proposed", 0)
        accepted = data.get("accepted", 0)
        # For DFlash, per-step accept rate = avg_commit/block_size. AL is the
        # mean commit length per draft step (more meaningful for DFlash).
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
            "budget": self.budget,
            "target_gguf": self.target_gguf,
            "draft_gguf": self.draft_gguf,
            "kv_type": self.kv_type,
        }
