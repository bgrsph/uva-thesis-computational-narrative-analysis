"""Tests for models/llama/compose_tcindep.py — the post-hoc composer that
derives temporal_causal_independent from temporal + causal Llama runs."""
import json

from models.llama import compose_tcindep


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _identity_fields():
    return {
        "wikidata_id": "1000000",
        "summary_id":  "en",
        "lang":        "en",
        "text":        "Alice arrived. Bob left.",
        "sentences":   ["Alice arrived. Bob left."],
        "events": [
            {"event_id": "e1", "sent_id": 0, "start": 0,  "end": 5,  "trigger": "Alice", "event_type": "ARRIVE"},
            {"event_id": "e2", "sent_id": 0, "start": 15, "end": 18, "trigger": "Bob",   "event_type": "DEPART"},
        ],
    }


def _llama_block(condition, relations_key, relations_list):
    """A minimal `condition_block` matching what infer_relations.py writes."""
    return {
        "source":          "llama",
        "model_id":        "meta-llama/Meta-Llama-3-8B-Instruct",
        "prompt_template": f"models/llama/prompts/{condition}.yaml",
        "prompt_rendered": "PROMPT",
        "response_raw":    json.dumps({relations_key: relations_list}),
        "response_parsed": {relations_key: relations_list},
        "parse_error":     None,
        "relations":       {relations_key: relations_list},
        "input_tokens":    10,
        "output_tokens":   20,
        "max_new_tokens":  8182,
        "hit_ctx_cap":     False,
    }


def test_compose_unions_temporal_and_causal_relations(tmp_path):
    temporal_path = tmp_path / "temporal.jsonl"
    causal_path   = tmp_path / "causal.jsonl"
    out_path      = tmp_path / "tcindep.jsonl"

    temporal_rel = [{"source": "e1", "target": "e2", "relation": "BEFORE"}]
    causal_rel   = [{"source": "e1", "target": "e2", "relation": "CAUSE"}]

    _write_jsonl(temporal_path, [{**_identity_fields(),
                                   "condition_block": _llama_block("temporal", "temporal_relations", temporal_rel)}])
    _write_jsonl(causal_path,   [{**_identity_fields(),
                                   "condition_block": _llama_block("causal",   "causal_relations",   causal_rel)}])

    compose_tcindep.main(temporal_path, causal_path, out_path)

    out_rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(out_rows) == 1

    # Identity carried through.
    assert out_rows[0]["wikidata_id"] == "1000000"
    assert out_rows[0]["events"][0]["event_id"] == "e1"

    cb = out_rows[0]["condition_block"]
    assert cb["source"]        == "composed"
    assert cb["composed_from"] == ["temporal", "causal"]
    assert cb["relations"]     == {
        "temporal_relations": temporal_rel,
        "causal_relations":   causal_rel,
    }
    # No Llama-only fields on a composed block.
    assert "prompt_rendered" not in cb
    assert "response_raw"    not in cb
    assert "input_tokens"    not in cb


def test_compose_inner_joins_dropping_keys_missing_in_either_side(tmp_path):
    """A summary present in `temporal` but absent in `causal` is dropped."""
    temporal_path = tmp_path / "temporal.jsonl"
    causal_path   = tmp_path / "causal.jsonl"
    out_path      = tmp_path / "tcindep.jsonl"

    row_a = {**_identity_fields(),  # wikidata_id 1000000 / summary_id "en"
             "condition_block": _llama_block("temporal", "temporal_relations",
                                             [{"source": "e1", "target": "e2", "relation": "BEFORE"}])}
    row_b = {**_identity_fields(),  # different key
             "wikidata_id": "2000000",
             "condition_block": _llama_block("temporal", "temporal_relations", [])}
    _write_jsonl(temporal_path, [row_a, row_b])

    # causal has only row_a's key, not row_b's.
    _write_jsonl(causal_path, [{**_identity_fields(),
                                "condition_block": _llama_block("causal", "causal_relations", [])}])

    compose_tcindep.main(temporal_path, causal_path, out_path)

    out_rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(out_rows) == 1
    assert out_rows[0]["wikidata_id"] == "1000000"   # row_b was dropped (no causal counterpart)


def test_compose_skips_keys_where_either_side_is_overflow(tmp_path):
    """If either temporal or causal has source='skipped_ctx_overflow' for a key,
    no composed row is written (tcindep is absent for that summary)."""
    temporal_path = tmp_path / "temporal.jsonl"
    causal_path   = tmp_path / "causal.jsonl"
    out_path      = tmp_path / "tcindep.jsonl"

    overflow_block = {
        "source":          "skipped_ctx_overflow",
        "model_id":        "meta-llama/Meta-Llama-3-8B-Instruct",
        "prompt_template": "models/llama/prompts/temporal.yaml",
        "prompt_rendered": "PROMPT",
        "response_raw":    None,
        "response_parsed": None,
        "parse_error":     None,
        "relations":       None,
        "input_tokens":    8192,
        "output_tokens":   None,
        "max_new_tokens":  None,
        "hit_ctx_cap":     None,
    }
    _write_jsonl(temporal_path, [{**_identity_fields(), "condition_block": overflow_block}])
    _write_jsonl(causal_path,   [{**_identity_fields(),
                                   "condition_block": _llama_block("causal", "causal_relations",
                                                                    [{"source": "e1", "target": "e2", "relation": "CAUSE"}])}])

    compose_tcindep.main(temporal_path, causal_path, out_path)

    # File exists but is empty — no composed row for the overflowed key.
    assert out_path.read_text() == ""


def test_compose_handles_parse_failure_rows_gracefully(tmp_path):
    """If one side has parse_error (relations == None), the composed row uses []."""
    temporal_path = tmp_path / "temporal.jsonl"
    causal_path   = tmp_path / "causal.jsonl"
    out_path      = tmp_path / "tcindep.jsonl"

    # Temporal succeeded, causal failed to parse.
    _write_jsonl(temporal_path, [{**_identity_fields(),
                                   "condition_block": _llama_block("temporal", "temporal_relations",
                                                                    [{"source": "e1", "target": "e2", "relation": "BEFORE"}])}])
    causal_failed_cb = _llama_block("causal", "causal_relations", [])
    causal_failed_cb["response_parsed"] = None
    causal_failed_cb["relations"]       = None
    causal_failed_cb["parse_error"]     = "JSONDecodeError: Expecting value: line 1 column 1 (char 0)"
    _write_jsonl(causal_path, [{**_identity_fields(), "condition_block": causal_failed_cb}])

    compose_tcindep.main(temporal_path, causal_path, out_path)

    out_rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(out_rows) == 1
    cb = out_rows[0]["condition_block"]
    assert cb["relations"]["temporal_relations"] == [{"source": "e1", "target": "e2", "relation": "BEFORE"}]
    assert cb["relations"]["causal_relations"]   == []
