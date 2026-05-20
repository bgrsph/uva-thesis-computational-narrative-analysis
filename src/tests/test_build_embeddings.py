"""Tests for the experiment.jsonl embeddings merger."""
import json
from pathlib import Path

from src import build_embeddings as merger


def _exp_row(wid: str, sid: str) -> dict:
    return {"wikidata_id": wid, "summary_id": sid, "text": "…", "embeddings": {}}


def _enc_row(wid: str, sid: str, encoder_key: str, dim: int = 4) -> dict:
    return {
        "wikidata_id": wid, "summary_id": sid,
        "encoder_key": encoder_key,
        "model_id":    f"vendor/{encoder_key}",
        "task":        None if encoder_key.startswith("sbert") else "Retrieve…",
        "dim":         dim,
        "vectors":     {k: [0.0] * dim for k in (
            "raw_text", "events_only", "temporal", "causal",
            "temporal_causal_independent", "temporal_causal_joint",
        )},
    }


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_merges_two_encoders_into_one_row(tmp_path):
    exp = tmp_path / "experiment.jsonl"
    e5  = tmp_path / "e5.jsonl"
    sb  = tmp_path / "sb.jsonl"
    _write_jsonl(exp, [_exp_row("w1", "en")])
    _write_jsonl(e5,  [_enc_row("w1", "en", "e5_mistral")])
    _write_jsonl(sb,  [_enc_row("w1", "en", "sbert_mpnet")])

    rc = merger.cli(["--in", str(exp), "--encoders", str(e5), str(sb), "--inplace"])
    assert rc == 0
    out = json.loads(exp.read_text().splitlines()[0])
    assert set(out["embeddings"].keys()) == {"e5_mistral", "sbert_mpnet"}
    assert out["embeddings"]["e5_mistral"]["dim"] == 4
    assert out["embeddings"]["e5_mistral"]["model_id"] == "vendor/e5_mistral"
    assert out["embeddings"]["sbert_mpnet"]["task"] is None
    assert set(out["embeddings"]["e5_mistral"]["vectors"].keys()) == {
        "raw_text", "events_only", "temporal", "causal",
        "temporal_causal_independent", "temporal_causal_joint",
    }


def test_re_run_replaces_encoder_slot(tmp_path):
    exp = tmp_path / "experiment.jsonl"
    e5  = tmp_path / "e5.jsonl"
    sb  = tmp_path / "sb.jsonl"
    _write_jsonl(exp, [_exp_row("w1", "en")])
    _write_jsonl(e5,  [_enc_row("w1", "en", "e5_mistral", dim=4)])
    _write_jsonl(sb,  [_enc_row("w1", "en", "sbert_mpnet", dim=4)])
    merger.cli(["--in", str(exp), "--encoders", str(e5), str(sb), "--inplace"])

    _write_jsonl(e5, [_enc_row("w1", "en", "e5_mistral", dim=8)])
    merger.cli(["--in", str(exp), "--encoders", str(e5), "--inplace"])

    out = json.loads(exp.read_text().splitlines()[0])
    assert out["embeddings"]["e5_mistral"]["dim"] == 8
    assert out["embeddings"]["sbert_mpnet"]["dim"] == 4


def test_missing_encoder_input_leaves_other_blocks(tmp_path):
    exp = tmp_path / "experiment.jsonl"
    e5  = tmp_path / "e5.jsonl"
    _write_jsonl(exp, [_exp_row("w1", "en"), _exp_row("w2", "en")])
    _write_jsonl(e5,  [_enc_row("w1", "en", "e5_mistral")])
    merger.cli(["--in", str(exp), "--encoders", str(e5), "--inplace"])

    lines = exp.read_text().splitlines()
    out = [json.loads(l) for l in lines]
    assert "e5_mistral" in out[0]["embeddings"]
    assert out[1]["embeddings"] == {}


def test_preserves_other_row_fields(tmp_path):
    exp = tmp_path / "experiment.jsonl"
    e5  = tmp_path / "e5.jsonl"
    row = _exp_row("w1", "en")
    row["text"] = "the story text"
    row["conditions"] = {"temporal": {"foo": "bar"}}
    _write_jsonl(exp, [row])
    _write_jsonl(e5,  [_enc_row("w1", "en", "e5_mistral")])
    merger.cli(["--in", str(exp), "--encoders", str(e5), "--inplace"])

    out = json.loads(exp.read_text().splitlines()[0])
    assert out["text"] == "the story text"
    assert out["conditions"] == {"temporal": {"foo": "bar"}}
