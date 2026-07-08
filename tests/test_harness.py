"""Unit tests for harness, labeling, and probe shapes.

Currently covers the experiment config only; harness and labeling tests are
added alongside their Phase 1 implementations.
"""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "experiment_config.yaml"


def load_config() -> dict:
    """Load the experiment config from its canonical location.

    Returns:
        The parsed YAML config as a dict.
    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_config_exists_and_parses() -> None:
    """The experiment config must exist and be valid YAML."""
    config = load_config()
    assert isinstance(config, dict)


def test_config_has_required_sections() -> None:
    """All top-level sections referenced by the pipeline must be present."""
    config = load_config()
    for section in ("model", "benchmark", "generation", "probing", "patching", "labels"):
        assert section in config, f"missing config section: {section}"


def test_label_taxonomy_is_complete() -> None:
    """The label set must be exactly the P/F1-F4 per-stage taxonomy."""
    config = load_config()
    assert config["labels"]["classes"] == ["P", "F1", "F2", "F3", "F4"]


def test_generation_settings_are_sane() -> None:
    """Generation settings must stay within the ranges the spike validated."""
    gen = load_config()["generation"]
    assert 0.0 <= gen["temperature"] <= 1.0
    assert gen["correction_rounds"] >= 1
    assert gen["max_new_tokens"] > 0
