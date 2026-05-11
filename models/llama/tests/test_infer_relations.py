"""CLI surface tests for models/llama/infer_relations.py.

No torch / transformers import. The script must expose a `parse_args` that
returns the parsed namespace so we can test it in isolation.
"""
import pytest

from models.llama import infer_relations as ir


def test_parse_args_minimum_required(tmp_path):
    args = ir.parse_args([
        "--input", str(tmp_path / "in.jsonl"),
        "--output", str(tmp_path / "out.jsonl"),
        "--condition", "causal",
    ])
    assert str(args.input).endswith("in.jsonl")
    assert str(args.output).endswith("out.jsonl")
    assert args.condition == "causal"
    assert args.sample_size is None


def test_parse_args_accepts_sample_size(tmp_path):
    args = ir.parse_args([
        "--input", str(tmp_path / "in.jsonl"),
        "--output", str(tmp_path / "out.jsonl"),
        "--condition", "temporal",
        "--sample-size", "3",
    ])
    assert args.sample_size == 3


@pytest.mark.parametrize("cond", [
    "temporal", "causal", "temporal_causal_independent", "temporal_causal_joint",
])
def test_parse_args_accepts_all_four_conditions(tmp_path, cond):
    args = ir.parse_args([
        "--input", str(tmp_path / "in.jsonl"),
        "--output", str(tmp_path / "out.jsonl"),
        "--condition", cond,
    ])
    assert args.condition == cond


def test_parse_args_rejects_unknown_condition(tmp_path):
    with pytest.raises(SystemExit):
        ir.parse_args([
            "--input", str(tmp_path / "in.jsonl"),
            "--output", str(tmp_path / "out.jsonl"),
            "--condition", "spatial",   # not a valid condition
        ])


def test_parse_args_requires_input_output_condition(tmp_path):
    # Missing --condition
    with pytest.raises(SystemExit):
        ir.parse_args([
            "--input", str(tmp_path / "in.jsonl"),
            "--output", str(tmp_path / "out.jsonl"),
        ])
