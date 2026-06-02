"""Unit tests for src/experiment_paths.py — the per-experiment folder namer (UNI-88)."""
import re

from src.experiment_paths import new_experiment_name


def test_auto_name_has_split_rows_and_timestamp():
    assert re.fullmatch(r"experiment_test_5057_\d{8}_\d{4}", new_experiment_name("test", 5057))


def test_explicit_timestamp_is_exact():
    assert new_experiment_name("test", 10, ts="20260101_0000") == "experiment_test_10_20260101_0000"
