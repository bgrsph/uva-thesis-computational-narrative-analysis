"""Canonical linearization of (events, relations) into one string per
experimental condition. See docs/superpowers/specs/2026-05-19-uni-26-linearization-design.md (UNI-26)."""
from __future__ import annotations

import copy


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


def linearize_relations(triples: list[dict], header: str) -> str:
    """Sort triples by (int(src[1:]), int(tgt[1:])) and format under <HEADER>:.
    Empty list still emits the header line."""
    parts = [f"{header}:\n"]
    for t in sorted(triples, key=lambda r: (int(r["source"][1:]), int(r["target"][1:]))):
        parts.append(f"({t['source']}, {t['relation']}, {t['target']})\n")
    return "".join(parts)


_CONDITION_SPEC: dict[str, tuple[tuple[str, str], ...]] = {
    # condition key in experiment.jsonl  -> tuple of (relations subkey, section header)
    "temporal":                    (("temporal_relations", "TEMPORAL"),),
    "causal":                      (("causal_relations",   "CAUSAL"),),
    "temporal_causal_joint":       (("joint_relations",    "TEMPEROCAUSAL"),),
    "temporal_causal_independent": (
        ("temporal_relations", "TEMPORAL"),
        ("causal_relations",   "CAUSAL"),
    ),
}


def linearize_row(row: dict) -> dict:
    """Return a new row dict with the 5 linearized strings populated.

    Does not mutate `row`. Reads relations from
    `conditions[<cond>]["relations"][<subkey>]` per `_CONDITION_SPEC`.
    A `parse_error != None` (or otherwise missing `relations`) on a condition
    is treated as an empty relation list — the section header is still emitted.
    """
    out = copy.deepcopy(row)
    events_line = linearize_events(out["events"])

    out["linearized_events_only"] = events_line

    for condition, sections in _CONDITION_SPEC.items():
        block = out["conditions"][condition]
        relations = block.get("relations") or {}
        parts = [events_line, "\n"]
        for subkey, header in sections:
            triples = relations.get(subkey) or []
            parts.append(linearize_relations(triples, header))
        block["linearized"] = "".join(parts)

    return out
