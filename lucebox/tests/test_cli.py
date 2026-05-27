import os

import lucebox.cli as cli
from lucebox.cli import app
from typer.testing import CliRunner


def test_benchmark_help_exposes_validation_depth_controls():
    result = CliRunner().invoke(app, ["benchmark", "--help"])

    assert result.exit_code == 0
    assert "--frontiers-repeat" in result.output
    assert "--agentic-repeat" in result.output
    assert "--agentic-session-turns" in result.output
    assert "--agentic-session-sessions" in result.output


def test_profile_help_exposes_snapshot_and_refresh_controls():
    result = CliRunner().invoke(app, ["profile", "--help"])

    assert result.exit_code == 0
    assert "--export-snapshot" in result.output
    assert "--force-refresh" in result.output
    assert "--dry-run" in result.output
    assert "--step" in result.output


def test_default_variant_honors_wrapper_env():
    old = os.environ.get("LUCEBOX_VARIANT")
    try:
        os.environ["LUCEBOX_VARIANT"] = "integration-props-uv-squared-clean-cuda12"

        assert cli._pick_variant_from_driver(555, "86") == (
            "integration-props-uv-squared-clean-cuda12"
        )
    finally:
        if old is None:
            os.environ.pop("LUCEBOX_VARIANT", None)
        else:
            os.environ["LUCEBOX_VARIANT"] = old
