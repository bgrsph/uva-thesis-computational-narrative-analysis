"""End-to-end tests for src/build_experiment.py — the merger that joins the
four per-condition JSONLs into the nested data/results/experiment.jsonl."""
import json

from src import build_experiment


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _identity():
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


def _composed_block(temporal_rel, causal_rel):
    return {
        "source":        "composed",
        "composed_from": ["temporal", "causal"],
        "relations": {
            "temporal_relations": temporal_rel,
            "causal_relations":   causal_rel,
        },
    }


def test_merge_produces_nested_row_with_four_condition_slots(tmp_path):
    temporal_path = tmp_path / "temporal.jsonl"
    causal_path   = tmp_path / "causal.jsonl"
    tcjoint_path  = tmp_path / "temporal_causal_joint.jsonl"
    tcindep_path  = tmp_path / "temporal_causal_independent.jsonl"
    out_path      = tmp_path / "experiment.jsonl"

    temporal_rel = [{"source": "e1", "target": "e2", "relation": "BEFORE"}]
    causal_rel   = [{"source": "e1", "target": "e2", "relation": "CAUSE"}]
    joint_rel    = [{"source": "e1", "target": "e2", "relation": "BEFORE_CAUSE"}]

    _write_jsonl(temporal_path, [{**_identity(), "condition_block": _llama_block("temporal", "temporal_relations", temporal_rel)}])
    _write_jsonl(causal_path,   [{**_identity(), "condition_block": _llama_block("causal",   "causal_relations",   causal_rel)}])
    _write_jsonl(tcjoint_path,  [{**_identity(), "condition_block": _llama_block("temporal_causal_joint", "joint_relations", joint_rel)}])
    _write_jsonl(tcindep_path,  [{**_identity(), "condition_block": _composed_block(temporal_rel, causal_rel)}])

    build_experiment.main(temporal_path, causal_path, tcjoint_path, tcindep_path, out_path)

    out_rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(out_rows) == 1

    row = out_rows[0]
    # Identity carried through.
    assert row["wikidata_id"] == "1000000"
    assert row["summary_id"]  == "en"
    assert row["events"][0]["event_id"] == "e1"

    # The four condition slots all present.
    assert set(row["conditions"].keys()) == {
        "temporal", "causal", "temporal_causal_joint", "temporal_causal_independent",
    }
    assert row["conditions"]["temporal"]["source"]                       == "llama"
    assert row["conditions"]["causal"]["source"]                         == "llama"
    assert row["conditions"]["temporal_causal_joint"]["source"]          == "llama"
    assert row["conditions"]["temporal_causal_independent"]["source"]    == "composed"
    assert row["conditions"]["temporal"]["relations"]["temporal_relations"] == temporal_rel
    assert row["conditions"]["temporal_causal_independent"]["relations"]["causal_relations"] == causal_rel

    # No condition_block wrapper at the top level (it's now nested under conditions).
    assert "condition_block" not in row

    # Reserved slots for follow-up stages.
    assert row["embeddings"] == {}
    assert row["baselines"]  == {}


def test_merge_handles_summaries_missing_from_some_conditions(tmp_path):
    """If a summary failed (or was skipped) in one condition, it still appears
    in experiment.jsonl with the surviving condition slots populated."""
    temporal_path = tmp_path / "temporal.jsonl"
    causal_path   = tmp_path / "causal.jsonl"
    tcjoint_path  = tmp_path / "temporal_causal_joint.jsonl"
    tcindep_path  = tmp_path / "temporal_causal_independent.jsonl"
    out_path      = tmp_path / "experiment.jsonl"

    row_a = {**_identity(),                       "condition_block": _llama_block("temporal", "temporal_relations", [])}
    row_b = {**_identity(), "wikidata_id": "2", "condition_block": _llama_block("temporal", "temporal_relations", [])}

    _write_jsonl(temporal_path, [row_a, row_b])
    # row_b absent from causal (e.g. ctx-overflow on that condition):
    _write_jsonl(causal_path,   [{**_identity(),                       "condition_block": _llama_block("causal", "causal_relations", [])}])
    _write_jsonl(tcjoint_path,  [{**_identity(),                       "condition_block": _llama_block("temporal_causal_joint", "joint_relations", [])},
                                  {**_identity(), "wikidata_id": "2", "condition_block": _llama_block("temporal_causal_joint", "joint_relations", [])}])
    _write_jsonl(tcindep_path,  [{**_identity(),                       "condition_block": _composed_block([], [])}])

    build_experiment.main(temporal_path, causal_path, tcjoint_path, tcindep_path, out_path)

    out_rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    by_wid = {r["wikidata_id"]: r for r in out_rows}
    assert set(by_wid.keys()) == {"1000000", "2"}

    # Full coverage for 1000000.
    assert set(by_wid["1000000"]["conditions"].keys()) == {
        "temporal", "causal", "temporal_causal_joint", "temporal_causal_independent",
    }
    # Partial for "2" — only temporal + joint had it; missing slots are simply absent.
    assert set(by_wid["2"]["conditions"].keys()) == {"temporal", "temporal_causal_joint"}


def test_merge_smoke_three_summaries_all_four_conditions(tmp_path):
    """UNI-65 acceptance #6: 3-row pilot through all conditions + composer + merger."""
    temporal_path = tmp_path / "temporal.jsonl"
    causal_path   = tmp_path / "causal.jsonl"
    tcjoint_path  = tmp_path / "temporal_causal_joint.jsonl"
    tcindep_path  = tmp_path / "temporal_causal_independent.jsonl"
    out_path      = tmp_path / "experiment.jsonl"

    def _row(wid):
        identity = _identity()
        identity["wikidata_id"] = wid
        return identity

    temporal_rows = [{**_row(w), "condition_block": _llama_block("temporal", "temporal_relations", [])} for w in ["1", "2", "3"]]
    causal_rows   = [{**_row(w), "condition_block": _llama_block("causal",   "causal_relations",   [])} for w in ["1", "2", "3"]]
    tcjoint_rows  = [{**_row(w), "condition_block": _llama_block("temporal_causal_joint", "joint_relations", [])} for w in ["1", "2", "3"]]
    tcindep_rows  = [{**_row(w), "condition_block": _composed_block([], [])} for w in ["1", "2", "3"]]
    _write_jsonl(temporal_path, temporal_rows)
    _write_jsonl(causal_path,   causal_rows)
    _write_jsonl(tcjoint_path,  tcjoint_rows)
    _write_jsonl(tcindep_path,  tcindep_rows)

    build_experiment.main(temporal_path, causal_path, tcjoint_path, tcindep_path, out_path)

    out_rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(out_rows) == 3
    for row in out_rows:
        assert set(row["conditions"].keys()) == {
            "temporal", "causal", "temporal_causal_joint", "temporal_causal_independent",
        }
        assert row["embeddings"] == {}
        assert row["baselines"]  == {}
