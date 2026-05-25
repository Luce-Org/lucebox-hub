"""lucebox — host-side CLI for the lucebox-hub container.

Runs inside the container; the host wrapper at ../lucebox.sh handles `docker
run` plumbing and systemd integration. This package owns: TOML config,
autotune rules, docker daemon calls (via the mounted socket), the benchmark
sweep orchestrator, the smoke test, and model download.
"""

__version__ = "0.2.0"
