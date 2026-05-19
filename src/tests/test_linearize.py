"""Unit tests for src/linearize.py — the UNI-26 canonical linearization templater."""
import json

import pytest

from src import linearize


def _event(event_id, sent_id, start, trigger, event_type, end=None):
    return {
        "event_id":   event_id,
        "sent_id":    sent_id,
        "start":      start,
        "end":        end if end is not None else start + len(trigger),
        "trigger":    trigger,
        "event_type": event_type,
    }


def test_event_unit_format():
    out = linearize.linearize_events([_event("e1", 0, 86, "enters", "Arriving")])
    assert out == "e1|enters|Arriving"


def test_event_sort_textual_order():
    # Out-of-order input; expect (sent_id ASC, start ASC).
    events = [
        _event("e3", 2, 10, "exists",   "Presence"),
        _event("e1", 0,  5, "enters",   "Arriving"),
        _event("e2", 0, 80, "finds",    "Know"),
    ]
    out = linearize.linearize_events(events)
    assert out == "e1|enters|Arriving e2|finds|Know e3|exists|Presence"


def test_pipe_in_trigger_fails_loud():
    bad = _event("e1", 0, 0, "a|b", "Arriving")
    with pytest.raises(AssertionError):
        linearize.linearize_events([bad])
