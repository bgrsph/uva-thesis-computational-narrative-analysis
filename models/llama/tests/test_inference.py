import json

import pytest

from models.llama.inference import (
    inline_events,
    load_prompt_config,
    parse_and_validate,
)


def test_inline_events_splices_markers_using_upstream_event_ids():
    sentences = [
        "The king died and the queen mourned.",  # sent 0
        "Then the prince ran away.",             # sent 1
    ]
    events = [  # upstream BERT+CRF output (event_ids already assigned)
        {"event_id": "e1", "sent_id": 0, "trigger": "died",    "event_type": "DEATH",   "start": 9,  "end": 13},
        {"event_id": "e2", "sent_id": 0, "trigger": "mourned", "event_type": "EMOTION", "start": 28, "end": 35},
        {"event_id": "e3", "sent_id": 1, "trigger": "ran",     "event_type": "MOTION",  "start": 16, "end": 19},
    ]

    out = inline_events(sentences, events)

    # Markers use upstream event_ids verbatim.
    assert "[e1|died|DEATH]" in out
    assert "[e2|mourned|EMOTION]" in out
    assert "[e3|ran|MOTION]" in out

    # Markers appear in document order.
    assert out.index("[e1|") < out.index("[e2|") < out.index("[e3|")

    # Non-event text outside spans is preserved byte-for-byte.
    assert "The king " in out and " and the queen " in out
    assert "Then the prince " in out and " away." in out


def test_parse_and_validate_happy_path_causal():
    cfg = load_prompt_config("causal")
    out_str = json.dumps({
        "causal_relations": [
            {"source": "e1", "target": "e3", "relation": "CAUSE"},
            {"source": "e2", "target": "e3", "relation": "ENABLE"},
        ],
    })

    parsed = parse_and_validate(out_str, cfg, "causal")

    assert len(parsed["causal_relations"]) == 2
    assert parsed["causal_relations"][0]["relation"] == "CAUSE"
    assert parsed["causal_relations"][1]["source"] == "e2"


@pytest.mark.parametrize(
    "out_str, expected_exc",
    [
        # Unknown relation label
        (
            json.dumps({"causal_relations": [
                {"source": "e1", "target": "e2", "relation": "INVENTED_LABEL"},
            ]}),
            ValueError,
        ),
        # Malformed event ID (not matching ^e\d+$)
        (
            json.dumps({"causal_relations": [
                {"source": "event-1", "target": "e2", "relation": "CAUSE"},
            ]}),
            ValueError,
        ),
        # Not valid JSON at all
        (
            "not valid json {",
            json.JSONDecodeError,
        ),
    ],
)
def test_parse_and_validate_rejects_bad_inputs(out_str, expected_exc):
    cfg = load_prompt_config("causal")
    with pytest.raises(expected_exc):
        parse_and_validate(out_str, cfg, "causal")


# ---- UNI-65: build_run_row ----

from models.llama.inference import build_run_row

_TEST_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def _events_input_row():
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


def test_build_run_row_happy_path():
    cfg = load_prompt_config("causal")
    valid_label = cfg["allowed_labels"][0]
    response_raw = json.dumps({"causal_relations": [
        {"source": "e1", "target": "e2", "relation": valid_label},
    ]})

    row = build_run_row(
        input_row=_events_input_row(),
        condition="causal",
        model_id=_TEST_MODEL_ID,
        prompt_rendered="PROMPT",
        response_raw=response_raw,
        input_tokens=10,
        output_tokens=42,
        max_new_tokens=8182,
        cfg=cfg,
    )

    # Identity / events preserved.
    assert row["wikidata_id"] == "1000000"
    assert row["summary_id"]  == "en"
    assert row["events"][0]["event_id"] == "e1"

    # condition_block fully populated on success.
    cb = row["condition_block"]
    assert cb["source"]           == "llama"
    assert cb["model_id"]         == _TEST_MODEL_ID
    assert cb["prompt_template"]  == "models/llama/prompts/causal.yaml"
    assert cb["prompt_rendered"]  == "PROMPT"
    assert cb["response_raw"]     == response_raw
    assert cb["response_parsed"]["causal_relations"][0]["relation"] == valid_label
    assert cb["parse_error"]      is None
    assert cb["relations"]        == cb["response_parsed"]
    assert cb["input_tokens"]     == 10
    assert cb["output_tokens"]    == 42
    assert cb["max_new_tokens"]   == 8182
    assert cb["hit_ctx_cap"]      is False    # 42 != 8182


def test_build_run_row_parse_failure_keeps_row():
    cfg = load_prompt_config("causal")
    response_raw = "not valid json {"

    row = build_run_row(
        input_row=_events_input_row(),
        condition="causal",
        model_id=_TEST_MODEL_ID,
        prompt_rendered="PROMPT",
        response_raw=response_raw,
        input_tokens=10,
        output_tokens=2048,
        max_new_tokens=2048,
        cfg=cfg,
    )

    # Row still produced.
    assert row["wikidata_id"] == "1000000"

    cb = row["condition_block"]
    assert cb["response_raw"]    == response_raw   # verbatim, preserved
    assert cb["response_parsed"] is None
    assert cb["relations"]       is None
    assert cb["parse_error"]     is not None
    assert "JSONDecodeError" in cb["parse_error"]
    assert cb["hit_ctx_cap"]     is True           # 2048 == 2048


def test_build_run_row_parse_failure_records_validation_error():
    cfg = load_prompt_config("causal")
    # JSON parses fine, but contains a label that's not in the causal codebook.
    response_raw = json.dumps({"causal_relations": [
        {"source": "e1", "target": "e2", "relation": "INVENTED_LABEL"},
    ]})

    row = build_run_row(
        input_row=_events_input_row(),
        condition="causal",
        model_id=_TEST_MODEL_ID,
        prompt_rendered="PROMPT",
        response_raw=response_raw,
        input_tokens=10,
        output_tokens=42,
        max_new_tokens=8182,
        cfg=cfg,
    )

    cb = row["condition_block"]
    assert cb["response_parsed"] is None
    assert cb["relations"]       is None
    assert cb["parse_error"].startswith("ValueError:")
    assert "INVENTED_LABEL" in cb["parse_error"]
