"""Canonical linearization of (events, relations) into one string per
experimental condition. See docs/superpowers/specs/2026-05-19-uni-26-linearization-design.md (UNI-26)."""
from __future__ import annotations


def linearize_events(events: list[dict]) -> str:
    """Format events in textual order as space-joined `eID|trigger|EVENT_TYPE` units."""
    units = []
    for ev in sorted(events, key=lambda e: (e["sent_id"], e["start"])):
        trigger = ev["trigger"]
        event_type = ev["event_type"]
        assert "|" not in trigger,    f"trigger contains '|': {trigger!r}"
        assert "|" not in event_type, f"event_type contains '|': {event_type!r}"
        units.append(f"{ev['event_id']}|{trigger}|{event_type}")
    return " ".join(units)
