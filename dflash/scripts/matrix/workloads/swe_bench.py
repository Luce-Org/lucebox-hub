"""SWE-bench Verified workload — wraps bench_agent.py helpers.

Reuses _load_swe_rows, select_rows_for_bucket, build_prompt, tokenize_to_file
from bench_agent.py so prompts are byte-identical to the existing DFlash bench.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Iterable, Dict, Any

ROOT = Path(__file__).resolve().parent.parent.parent.parent   # dflash/
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from bench_agent import (   # noqa: E402
    BUCKETS,
    build_prompt,
    tokenize_to_file,
    _load_swe_rows,
    select_rows_for_bucket,
)

from matrix.workload import Workload, WorkloadPrompt   # noqa: E402


class SweBenchWorkload(Workload):
    """SWE-bench Verified agentic prompt workload.

    Supports three bucket sizes that match bench_agent.py:
        2k  (~2048 tokens)
        8k  (~8192 tokens)
        24k (~24576 tokens)

    Parameters
    ----------
    bucket:
        One of "2k", "8k", "24k".
    n_sample:
        Number of SWE rows to sample.
    seed:
        Random seed for row selection (default 42).
    tokenizer_id:
        HuggingFace tokenizer identifier.
    tmpdir:
        Where to write tokenised .bin files (defaults to system tmp).
    """

    def __init__(
        self,
        bucket: str = "2k",
        n_sample: int = 8,
        seed: int = 42,
        tokenizer_id: str = "Qwen/Qwen3.5-27B",
        tmpdir: Path | None = None,
    ) -> None:
        if bucket not in BUCKETS:
            raise ValueError(f"bucket must be one of {list(BUCKETS)}, got {bucket!r}")
        self.bucket = bucket
        self.n_sample = n_sample
        self.seed = seed
        self.tokenizer_id = tokenizer_id
        self.tmpdir = tmpdir or Path(tempfile.gettempdir()) / "dflash_bench_matrix"
        self.tmpdir.mkdir(parents=True, exist_ok=True)
        self.name = f"swe_bench_{bucket}"

    def prompts(self) -> Iterable[WorkloadPrompt]:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.tokenizer_id, trust_remote_code=True)

        cfg = BUCKETS[self.bucket]
        target_tokens = cfg["target"]
        sys_prompt_path = cfg["sys"]

        df = _load_swe_rows()
        rows = select_rows_for_bucket(df, target_tokens, self.n_sample, seed=self.seed)

        for idx, row in enumerate(rows):
            text, n_tokens = build_prompt(tok, sys_prompt_path, row, target_tokens)
            bin_path = self.tmpdir / f"swe_{self.bucket}_{idx:03d}.bin"
            token_ids = tok.encode(text, add_special_tokens=False)
            tokenize_to_file(tok, text, bin_path)

            wp = WorkloadPrompt.from_tokens(
                idx=idx,
                prompt_id=row.get("instance_id", f"row_{idx}"),
                token_ids=token_ids,
            )
            # Attach the path so callers can pass it directly to speculators.
            wp.bin_path = bin_path   # type: ignore[attr-defined]
            yield wp

    def config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "bucket": self.bucket,
            "n_sample": self.n_sample,
            "seed": self.seed,
            "tokenizer_id": self.tokenizer_id,
        }
