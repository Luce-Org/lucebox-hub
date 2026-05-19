"""Abstract Workload base class for bench_matrix."""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Iterable


@dataclass
class WorkloadPrompt:
    """A single prompt ready for benchmarking."""
    idx: int
    prompt_id: str          # stable identifier (e.g. SWE instance_id)
    prompt_tokens: list     # list[int] token ids
    n_prompt_tokens: int
    sha256: str             # sha256 hex of the raw token bytes — for reproducibility checks

    @classmethod
    def from_tokens(cls, idx: int, prompt_id: str, token_ids: list) -> "WorkloadPrompt":
        raw = b"".join(struct.pack("<i", t) for t in token_ids)
        sha = hashlib.sha256(raw).hexdigest()
        return cls(
            idx=idx,
            prompt_id=prompt_id,
            prompt_tokens=token_ids,
            n_prompt_tokens=len(token_ids),
            sha256=sha,
        )

    def write_bin(self, path: Path) -> Path:
        """Write token ids as little-endian int32 binary file (test_dflash format)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            for t in self.prompt_tokens:
                f.write(struct.pack("<i", int(t)))
        return path


class Workload:
    """Abstract base: yields WorkloadPrompts, optionally grades outputs.

    Subclasses MUST set ``name`` and implement ``prompts()``.
    """

    name: str = "abstract"
    seed: int = 42
    n_sample: int = 8

    def prompts(self) -> Iterable[WorkloadPrompt]:
        raise NotImplementedError(f"{type(self).__name__}.prompts() not implemented")

    def grade(
        self,
        prompt: WorkloadPrompt,
        generated_tokens: list,
    ) -> Dict[str, Any]:
        """Optional quality grader. Default: count tokens generated."""
        return {"tokens_generated": len(generated_tokens)}

    def config(self) -> Dict[str, Any]:
        """Serialisable config for artifact metadata."""
        return {
            "name": self.name,
            "seed": self.seed,
            "n_sample": self.n_sample,
        }
