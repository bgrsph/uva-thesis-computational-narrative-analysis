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


# ---- UNI-74: section-mismatch recovery for temporal_causal_independent ----


def test_parse_and_validate_reroutes_causal_label_from_temporal_section():
    """CAUSE_TO_END placed under temporal_relations should be moved to causal_relations."""
    cfg = load_prompt_config("temporal_causal_independent")
    out_str = json.dumps({
        "temporal_relations": [
            {"source": "e1", "target": "e2", "relation": "BEFORE"},        # correctly placed
            {"source": "e3", "target": "e4", "relation": "CAUSE_TO_END"},  # misplaced
        ],
        "causal_relations": [],
    })

    parsed = parse_and_validate(out_str, cfg, "temporal_causal_independent")

    # The temporal relation stays where it was.
    assert parsed["temporal_relations"] == [
        {"source": "e1", "target": "e2", "relation": "BEFORE"},
    ]
    # The causal label was rerouted, not rejected.
    assert parsed["causal_relations"] == [
        {"source": "e3", "target": "e4", "relation": "CAUSE_TO_END"},
    ]


def test_parse_and_validate_reroutes_temporal_label_from_causal_section():
    """BEFORE placed under causal_relations should be moved to temporal_relations."""
    cfg = load_prompt_config("temporal_causal_independent")
    out_str = json.dumps({
        "temporal_relations": [],
        "causal_relations": [
            {"source": "e1", "target": "e2", "relation": "BEFORE"},   # misplaced
            {"source": "e3", "target": "e4", "relation": "CAUSE"},    # correctly placed
        ],
    })

    parsed = parse_and_validate(out_str, cfg, "temporal_causal_independent")
    assert parsed["temporal_relations"] == [
        {"source": "e1", "target": "e2", "relation": "BEFORE"},
    ]
    assert parsed["causal_relations"] == [
        {"source": "e3", "target": "e4", "relation": "CAUSE"},
    ]


def test_parse_and_validate_reroutes_both_directions_simultaneously():
    """Two misplaced relations crossing in opposite directions both recover."""
    cfg = load_prompt_config("temporal_causal_independent")
    out_str = json.dumps({
        "temporal_relations": [{"source": "e1", "target": "e2", "relation": "ENABLE"}],
        "causal_relations":   [{"source": "e3", "target": "e4", "relation": "OVERLAPS"}],
    })
    parsed = parse_and_validate(out_str, cfg, "temporal_causal_independent")
    assert parsed["temporal_relations"] == [
        {"source": "e3", "target": "e4", "relation": "OVERLAPS"},
    ]
    assert parsed["causal_relations"] == [
        {"source": "e1", "target": "e2", "relation": "ENABLE"},
    ]


def test_parse_and_validate_still_raises_on_true_hallucination_in_independent():
    """A label that's in NEITHER section's codebook still raises (no auto-recovery)."""
    cfg = load_prompt_config("temporal_causal_independent")
    out_str = json.dumps({
        "temporal_relations": [
            {"source": "e1", "target": "e2", "relation": "AFTER"},   # in neither codebook
        ],
        "causal_relations": [],
    })
    with pytest.raises(ValueError, match="unknown label: AFTER"):
        parse_and_validate(out_str, cfg, "temporal_causal_independent")


def test_parse_and_validate_single_section_condition_unchanged():
    """For conditions with one section (e.g. causal), there's no other section
    to move to, so a misplaced label still raises — old behavior preserved."""
    cfg = load_prompt_config("causal")
    out_str = json.dumps({
        "causal_relations": [
            {"source": "e1", "target": "e2", "relation": "BEFORE"},   # temporal label, no other section
        ],
    })
    with pytest.raises(ValueError, match="unknown label: BEFORE"):
        parse_and_validate(out_str, cfg, "causal")
