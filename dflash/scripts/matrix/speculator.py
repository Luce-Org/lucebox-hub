"""Abstract Speculator base class for bench_matrix."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class SpeculatorResult:
    """Parsed result from a single test_dflash invocation."""
    tok_s: float
    accept_rate: Optional[float]      # None for pure-AR (gamma=0)
    al: Optional[float]               # accept-length per outer iter (DFlash AL metric)
    n_tokens_generated: int
    decode_s: float
    prefill_s: float
    raw_json: Dict[str, Any]          # untouched parsed data for archival


# Regex that matches the RESULT line emitted by test_dflash / bench_agent_mtp.
# Format:  RESULT tok_s=X prompt=X gamma=X tokens=X decode_s=X prefill_s=X accepted=X proposed=X
RESULT_RE = re.compile(
    r"RESULT tok_s=(\S+) prompt=(\S+) gamma=(\S+) tokens=(\S+) "
    r"decode_s=(\S+) prefill_s=(\S+) accepted=(\S+) proposed=(\S+)"
)

# Alternate: RESULT_JSON line (experiment-C wiring)
RESULT_JSON_RE = re.compile(r"^RESULT_JSON (.+)$", re.MULTILINE)


def parse_result_line(stdout: str) -> Dict[str, Any]:
    """Parse the RESULT / RESULT_JSON line(s) from test_dflash stdout.

    Returns a dict with at minimum:
        tok_s, gamma, tokens, decode_s, prefill_s, accepted, proposed
    If a RESULT_JSON line is present its contents are merged in (they win).
    """
    m = RESULT_RE.search(stdout)
    if not m:
        raise ValueError(f"No RESULT line found in test_dflash output:\n{stdout[-2000:]}")

    tok_s, prompt, gamma, tokens, decode_s, prefill_s, accepted, proposed = m.groups()
    result: Dict[str, Any] = {
        "tok_s": float(tok_s),
        "prompt": int(prompt),
        "gamma": int(gamma),
        "tokens": int(tokens),
        "decode_s": float(decode_s),
        "prefill_s": float(prefill_s),
        "accepted": int(accepted),
        "proposed": int(proposed),
    }
    proposed_i = int(proposed)
    result["accept_rate"] = (int(accepted) / proposed_i) if proposed_i > 0 else None

    # Merge RESULT_JSON if present — it carries extra fields (al, ddtree_*, …)
    import json
    mj = RESULT_JSON_RE.search(stdout)
    if mj:
        try:
            extra = json.loads(mj.group(1))
            result.update(extra)
        except Exception:
            pass

    return result


class Speculator:
    """Abstract base: launches test_dflash with appropriate flags, parses RESULT.

    Subclasses MUST set ``name`` and implement ``run()`` and ``config()``.
    """

    name: str = "abstract"

    def run(
        self,
        prompt_path: str,
        prompt_id: int,
        n_prompt: int,
        n_gen: int,
        max_ctx: int,
    ) -> SpeculatorResult:
        raise NotImplementedError(f"{type(self).__name__}.run() not implemented")

    def config(self) -> Dict[str, Any]:
        """Serialisable config for artifact metadata."""
        return {"name": self.name}

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_ctx_for(n_prompt: int, n_gen: int) -> int:
        need = n_prompt + n_gen + 128
        for cap in (2048, 4096, 8192, 16384, 32768, 65536):
            if cap >= need:
                return cap
        return need

    @staticmethod
    def _make_env() -> Dict[str, str]:
        """Minimal clean env for subprocess (WSL2 CUDA flake mitigation)."""
        import os
        return {
            "HOME": os.environ.get("HOME", ""),
            "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
            "LD_LIBRARY_PATH": "/usr/lib/wsl/lib:/usr/local/cuda/lib64",
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        }
