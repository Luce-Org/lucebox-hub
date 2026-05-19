"""HumanEval workload — loads openai_humaneval test split.

Mirrors swe_bench.py pattern: dataset load, chat-template wrap,
tokenize_to_file, yield WorkloadPrompts. Throughput-only (no grading).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Iterable, Dict, Any

ROOT = Path(__file__).resolve().parent.parent.parent.parent   # dflash/
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from bench_agent import tokenize_to_file   # noqa: E402
from matrix.workload import Workload, WorkloadPrompt   # noqa: E402

# Dataset spec — mirrors bench_llm.py line 42.
_DS_NAME = "openai_humaneval"
_DS_CFG = None
_DS_SPLIT = "test"
_EXTRACT = lambda x: x["prompt"]   # noqa: E731
_N_GEN_DEFAULT = 256


class HumanEvalWorkload(Workload):
    """HumanEval workload (openai_humaneval test split).

    Parameters
    ----------
    n_sample:
        Number of problems to sample.
    seed:
        Shuffle seed for reproducibility (default 42).
    n_gen:
        Hint stored in config; the matrix runner controls actual n_gen.
    tokenizer_id:
        HuggingFace tokenizer identifier.
    tmpdir:
        Where to write tokenised .bin files (defaults to system tmp).
    """

    name = "humaneval"

    def __init__(
        self,
        n_sample: int = 8,
        seed: int = 42,
        n_gen: int = _N_GEN_DEFAULT,
        tokenizer_id: str = "Qwen/Qwen3.5-27B",
        tmpdir: Path | None = None,
    ) -> None:
        self.n_sample = n_sample
        self.seed = seed
        self.n_gen = n_gen
        self.tokenizer_id = tokenizer_id
        self.tmpdir = tmpdir or Path(tempfile.gettempdir()) / "dflash_bench_matrix"
        self.tmpdir.mkdir(parents=True, exist_ok=True)

    def prompts(self) -> Iterable[WorkloadPrompt]:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.tokenizer_id, trust_remote_code=True)

        ds = load_dataset(_DS_NAME, _DS_CFG, split=_DS_SPLIT)
        ds_sel = ds.shuffle(seed=self.seed).select(range(min(self.n_sample, len(ds))))

        for idx, sample in enumerate(ds_sel):
            raw_prompt = _EXTRACT(sample)
            task_id = sample.get("task_id", f"humaneval_{idx}")

            try:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": raw_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception:
                text = raw_prompt

            bin_path = self.tmpdir / f"humaneval_{idx:03d}.bin"
            tokenize_to_file(tok, text, bin_path)
            token_ids = tok.encode(text, add_special_tokens=False)

            wp = WorkloadPrompt.from_tokens(
                idx=idx,
                prompt_id=str(task_id),
                token_ids=token_ids,
            )
            wp.bin_path = bin_path   # type: ignore[attr-defined]
            yield wp

    def config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dataset": _DS_NAME,
            "split": _DS_SPLIT,
            "n_sample": self.n_sample,
            "seed": self.seed,
            "n_gen_hint": self.n_gen,
            "tokenizer_id": self.tokenizer_id,
        }
