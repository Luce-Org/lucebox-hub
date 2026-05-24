"""Vendored copy of forge-guardrails eval harness (tests/eval tree).

The forge-guardrails PyPI wheel only ships ``src/forge/`` — the eval
scenarios, ablation presets, and ``run_eval`` driver live under
``tests/eval/`` and are NOT installed by ``pip install forge-guardrails``.
We mirror the subset we need here so ``bench_http_capability.py --area
forge`` can drive forge's tool-calling scenarios against a self-hosted
dflash_server without a runtime download.

Source: https://github.com/antoinezambelli/forge/tree/main/tests/eval
Vendored from forge-guardrails 0.7.1.

Local modifications:
 - ``tests.eval.ablation`` imports rewritten to relative
   (``from .ablation import ...``).
 - ``tests.eval.scenarios`` imports rewritten to relative
   (``from .scenarios import ...``).
 - The inline ``from tests.eval.batch_eval import _compute_cost`` import
   inside ``run_eval`` is replaced with a stub that returns 0.0; the
   real pricing table is Anthropic-API-only and not meaningful for a
   self-hosted dflash bench.
 - ``main()`` (forge's standalone CLI) is dropped — bench_http_capability
   owns the entrypoint and only needs ``run_eval`` / ``RunResult`` /
   ``EvalConfig``.

Keep this tree updated when bumping the forge-guardrails dep; the
scenario set may drift across minor versions.
"""
