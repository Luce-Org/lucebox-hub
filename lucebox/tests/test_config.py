from lucebox import config


def test_legacy_env_migration_skips_invalid_values(tmp_path):
    legacy = tmp_path / "config.env"
    legacy.write_text(
        "DFLASH_BUDGET=not-an-int\n"
        "DFLASH_MAX_CTX=65536\n"
        "DFLASH_LAZY=true\n"
    )

    cfg = config._load_legacy_env(legacy)

    assert cfg.dflash.budget == 22
    assert cfg.dflash.max_ctx == 65536
    assert cfg.dflash.lazy is True


def test_image_variant_round_trips_from_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "[image]\n"
        'registry = "ghcr.io/luce-org/lucebox-hub"\n'
        'variant = "integration-props-uv-squared-clean-cuda12"\n'
    )

    cfg = config._load_toml(path)

    assert cfg.image == "ghcr.io/luce-org/lucebox-hub"
    assert cfg.variant == "integration-props-uv-squared-clean-cuda12"
