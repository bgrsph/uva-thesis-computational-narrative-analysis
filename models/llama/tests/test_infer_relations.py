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
    "temporal", "causal", "temporal_causal_joint",
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


# -- run_inference loop tests (monkeypatched pipeline; no torch/transformers load) ----

import json

from models.llama.inference import load_prompt_config


def _fake_tokenizer():
    class T:
        eos_token_id = 0
        def convert_tokens_to_ids(self, _tok):
            return 1
        def apply_chat_template(self, _msgs, tokenize=False, add_generation_prompt=True):
            return "PROMPT"
        def encode(self, _s, add_special_tokens=False):
            return [0] * 7   # deterministic token count
    return T()


def _make_fake_pipeline(canned_output: str):
    class P:
        tokenizer = _fake_tokenizer()
        def __call__(self, messages, **kw):
            return [{"generated_text": [{"role": "assistant", "content": canned_output}]}]
    return P()


def _write_events_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _events_row():
    # Sentence offsets are sentence-local (per inline_events impl).
    # "Alice arrived. Bob left."   "Alice" -> [0,5)   "Bob" -> [15,18)
    return {
        "sentences": ["Alice arrived. Bob left."],
        "events": [
            {"event_id": "e1", "sent_id": 0, "start": 0,  "end": 5,  "trigger": "Alice", "event_type": "ARRIVE"},
            {"event_id": "e2", "sent_id": 0, "start": 15, "end": 18, "trigger": "Bob",   "event_type": "DEPART"},
        ],
    }


def test_run_inference_writes_one_output_row_per_valid_input(tmp_path, monkeypatch):
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_events_jsonl(in_path, [_events_row(), _events_row()])

    cfg = load_prompt_config("causal")
    valid_label = cfg["allowed_labels"][0]
    canned = json.dumps({"causal_relations": [{"source": "e1", "target": "e2", "relation": valid_label}]})

    monkeypatch.setattr(ir, "_build_pipeline", lambda _model_id: _make_fake_pipeline(canned))

    args = ir.parse_args([
        "--input", str(in_path), "--output", str(out_path),
        "--condition", "causal", "--sample-size", "2",
    ])
    rc = ir.run_inference(args)
    assert rc == 0

    out_rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(out_rows) == 2
    for row in out_rows:
        assert "relations" in row
        assert row["relations"]["causal_relations"][0]["relation"] == valid_label


def test_run_inference_skips_rows_when_input_exceeds_ctx(tmp_path, monkeypatch, capsys):
    """If tokenized input is >= LLAMA_CTX, the row is skipped before pipe() is called."""
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_events_jsonl(in_path, [_events_row()])

    # Fake tokenizer that returns >= LLAMA_CTX tokens for the input.
    class _OverflowTokenizer:
        eos_token_id = 0
        def convert_tokens_to_ids(self, _tok): return 1
        def apply_chat_template(self, _msgs, tokenize=False, add_generation_prompt=True): return "PROMPT"
        def encode(self, _s, add_special_tokens=False): return [0] * ir.LLAMA_CTX  # exactly at the cap

    class _PipelineCalled(Exception): pass
    class _NeverCallPipeline:
        tokenizer = _OverflowTokenizer()
        def __call__(self, *_a, **_kw): raise _PipelineCalled("pipe() must not be invoked on oversize input")

    monkeypatch.setattr(ir, "_build_pipeline", lambda _model_id: _NeverCallPipeline())

    args = ir.parse_args([
        "--input", str(in_path), "--output", str(out_path), "--condition", "causal",
    ])
    rc = ir.run_inference(args)
    assert rc == 0
    assert out_path.read_text() == ""
    captured = capsys.readouterr()
    assert "exceeds ctx" in captured.out


def test_run_inference_skips_rows_on_validator_failure(tmp_path, monkeypatch, capsys):
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_events_jsonl(in_path, [_events_row()])

    # Canned output is not valid JSON -> JSONDecodeError -> row skipped.
    monkeypatch.setattr(ir, "_build_pipeline", lambda _model_id: _make_fake_pipeline("not-json{"))

    args = ir.parse_args([
        "--input", str(in_path), "--output", str(out_path), "--condition", "causal",
    ])
    rc = ir.run_inference(args)
    assert rc == 0
    assert out_path.read_text() == ""   # the single bad row was skipped, output is empty
    captured = capsys.readouterr()
    assert "row 0" in captured.out
