from models.llama.inference import inline_events


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
