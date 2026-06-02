"""Driver-loop tests. No real model weights — fake SentenceTransformer."""
import json

import numpy as np
import pytest

from models.embed import encode_embeddings as drv


class FakeSentenceTransformer:
    """Returns a fixed-shape ndarray; ignores the texts apart from their count."""
    def __init__(self, dim: int = 4):
        self._dim = dim

    def get_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, **kw):
        n = len(texts)
        out = np.zeros((n, self._dim), dtype=np.float32)
        out[:, 0] = 1.0   # unit vector along first axis
        return out


def _make_row(wid: str, sid: str) -> dict:
    return {
        "wikidata_id": wid,
        "summary_id":  sid,
        "text":        "Once upon a time…",
        "linearized_events_only": "EVENTS:\n(e1|enters|Arriving)\n",
        "conditions": {
            "temporal":                    {"linearized": "EVENTS:\n(e1|enters|Arriving)\nTEMPORAL:\n"},
            "causal":                      {"linearized": "EVENTS:\n(e1|enters|Arriving)\nCAUSAL:\n"},
            "temporal_causal_independent": {"linearized": "EVENTS:\n(e1|enters|Arriving)\nTEMPORAL:\nCAUSAL:\n"},
            "temporal_causal_joint":       {"linearized": "EVENTS:\n(e1|enters|Arriving)\nTEMPEROCAUSAL:\n"},
        },
    }


def test_six_condition_keys_in_fixed_order():
    assert drv.CONDITION_KEYS == (
        "raw_text", "events_only",
        "temporal", "causal",
        "temporal_causal_independent", "temporal_causal_joint",
    )


def test_writes_one_row_per_input(tmp_path, monkeypatch):
    in_path = tmp_path / "experiment.jsonl"
    out_path = tmp_path / "e5_mistral.jsonl"
    with in_path.open("w") as f:
        for r in (_make_row("w1", "en"), _make_row("w2", "de")):
            f.write(json.dumps(r) + "\n")

    monkeypatch.setattr(drv, "_build_model", lambda mid: FakeSentenceTransformer(dim=4))

    rc = drv.main([
        "--encoder", "e5_mistral",
        "--input",   str(in_path),
        "--output",  str(out_path),
    ])
    assert rc == 0

    rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(rows) == 2
    for r in rows:
        assert r["encoder_key"] == "e5_mistral"
        assert r["model_id"]    == "intfloat/e5-mistral-7b-instruct"
        assert r["task"]        == "Retrieve stories with a similar narrative to the given story."
        assert r["dim"]         == 4
        assert tuple(r["vectors"].keys()) == drv.CONDITION_KEYS
        for vec in r["vectors"].values():
            assert isinstance(vec, list) and len(vec) == 4


def test_sbert_writes_null_task(tmp_path, monkeypatch):
    in_path = tmp_path / "experiment.jsonl"
    out_path = tmp_path / "sbert.jsonl"
    in_path.write_text(json.dumps(_make_row("w1", "en")) + "\n")
    monkeypatch.setattr(drv, "_build_model", lambda mid: FakeSentenceTransformer(dim=4))

    drv.main([
        "--encoder", "sbert_mpnet",
        "--input",   str(in_path),
        "--output",  str(out_path),
    ])

    row = json.loads(out_path.read_text().splitlines()[0])
    assert row["encoder_key"] == "sbert_mpnet"
    assert row["task"] is None


def test_story_emb_writes_raw_text_only(tmp_path, monkeypatch):
    """story_emb's ENCODER_CONDITIONS override restricts output to {"raw_text"}."""
    in_path = tmp_path / "experiment.jsonl"
    out_path = tmp_path / "story_emb.jsonl"
    with in_path.open("w") as f:
        for r in (_make_row("w1", "en"), _make_row("w2", "de")):
            f.write(json.dumps(r) + "\n")

    monkeypatch.setattr(drv, "_build_model", lambda mid: FakeSentenceTransformer(dim=4))

    rc = drv.main([
        "--encoder", "story_emb",
        "--input",   str(in_path),
        "--output",  str(out_path),
    ])
    assert rc == 0

    rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(rows) == 2
    for r in rows:
        assert r["encoder_key"] == "story_emb"
        assert r["model_id"]    == "uhhlt/story-emb"
        assert r["task"]        == "Retrieve stories with a similar narrative to the given story."
        assert r["dim"]         == 4
        # The only condition embedded for story_emb is raw_text — no events_only, no temporal, etc.
        assert set(r["vectors"].keys()) == {"raw_text"}
        assert isinstance(r["vectors"]["raw_text"], list) and len(r["vectors"]["raw_text"]) == 4


def test_qwen3_emb_0p6b_forwards_init_kwargs(tmp_path, monkeypatch):
    """The qwen3 entry in ENCODER_INIT_KWARGS must reach SentenceTransformer()."""
    in_path = tmp_path / "experiment.jsonl"
    out_path = tmp_path / "qwen3_emb_0p6b.jsonl"
    in_path.write_text(json.dumps(_make_row("w1", "en")) + "\n")

    seen: dict = {}

    def fake_build(model_id: str, **kw):
        seen["model_id"] = model_id
        seen["kwargs"] = kw
        return FakeSentenceTransformer(dim=4)

    monkeypatch.setattr(drv, "_build_model", fake_build)

    drv.main([
        "--encoder", "qwen3_emb_0p6b",
        "--input",   str(in_path),
        "--output",  str(out_path),
    ])

    assert seen["model_id"] == "Qwen/Qwen3-Embedding-0.6B"
    assert seen["kwargs"]   == {"processor_kwargs": {"padding_side": "left"}}


def test_killed_job_leaves_clean_prefix(tmp_path, monkeypatch):
    """A fault midway through encoding leaves the prefix of completed rows intact."""
    in_path = tmp_path / "experiment.jsonl"
    out_path = tmp_path / "out.jsonl"
    with in_path.open("w") as f:
        for i in range(3):
            f.write(json.dumps(_make_row(f"w{i}", "en")) + "\n")

    class FlakyFake(FakeSentenceTransformer):
        def __init__(self):
            super().__init__(dim=4)
            self.calls = 0

        def encode(self, texts, **kw):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated CUDA OOM mid-run")
            return super().encode(texts, **kw)

    monkeypatch.setattr(drv, "_build_model", lambda mid: FlakyFake())

    with pytest.raises(RuntimeError, match="simulated CUDA OOM"):
        drv.main([
            "--encoder", "sbert_mpnet",
            "--input",   str(in_path),
            "--output",  str(out_path),
        ])

    lines = out_path.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])   # parses cleanly — no truncated tail
    assert row["wikidata_id"] == "w0"
