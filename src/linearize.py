"""Canonical linearization of (events, relations) into one string per
experimental condition. See docs/superpowers/specs/2026-05-19-uni-26-linearization-design.md (UNI-26)."""
from __future__ import annotations

import copy


def linearize_events(events: list[dict]) -> str:
    """Format events in textual order as space-joined `(eID|trigger|EVENT_TYPE)` units.
    Parens make unit boundaries unambiguous even when triggers contain spaces."""
    units = []
    for ev in sorted(events, key=lambda e: (e["sent_id"], e["start"])):
        trigger = ev["trigger"]
        event_type = ev["event_type"]
        assert "|" not in trigger,    f"trigger contains '|': {trigger!r}"
        assert "|" not in event_type, f"event_type contains '|': {event_type!r}"
        units.append(f"({ev['event_id']}|{trigger}|{event_type})")
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
    events_block = f"EVENTS:\n{linearize_events(out['events'])}\n"

    out["linearized_events_only"] = events_block

    for condition, sections in _CONDITION_SPEC.items():
        block = out["conditions"][condition]
        relations = block.get("relations") or {}
        parts = [events_block]
        for subkey, header in sections:
            triples = relations.get(subkey) or []
            parts.append(linearize_relations(triples, header))
        block["linearized"] = "".join(parts)

    return out


# --- Hybrid variant: relation endpoints carry their surface form -------------
# The original `linearize_relations` renders endpoints as bare ids, e.g.
# `(e2, BEFORE, e1)`. The hybrid form below renders each endpoint as
# `eID:trigger|EVENT_TYPE`, e.g. `(e2:left|Departing, BEFORE, e1:arrived|Arriving)`,
# so the embedder sees the surface trigger and type instead of an opaque pointer,
# while the id still disambiguates repeated triggers. The EVENTS block and the
# events-only condition are unchanged.

def _event_lookup(events: list[dict]) -> dict:
    """Map event_id -> (trigger, event_type) for hybrid relation linearization."""
    return {ev["event_id"]: (ev["trigger"], ev["event_type"]) for ev in events}


def linearize_relations_hybrid(triples: list[dict], header: str,
                               event_lookup: dict) -> str:
    """Like `linearize_relations`, but each endpoint is rendered as
    `eID:trigger|EVENT_TYPE` instead of the bare `eID`. Endpoints missing from
    `event_lookup` (e.g. an id the LLM emitted that is not in `events`) fall back
    to the bare id. Sorting and empty-list behaviour match `linearize_relations`."""
    def render(eid: str) -> str:
        info = event_lookup.get(eid)
        if info is None:
            return eid
        trigger, event_type = info
        return f"{eid}:{trigger}|{event_type}"

    parts = [f"{header}:\n"]
    for t in sorted(triples, key=lambda r: (int(r["source"][1:]), int(r["target"][1:]))):
        parts.append(f"({render(t['source'])}, {t['relation']}, {render(t['target'])})\n")
    return "".join(parts)


def linearize_row_hybrid(row: dict) -> dict:
    """Hybrid counterpart of `linearize_row`: relation endpoints carry their
    trigger and event type (`eID:trigger|EVENT_TYPE`). The EVENTS block and the
    events-only condition are identical to `linearize_row`. Does not mutate `row`."""
    out = copy.deepcopy(row)
    events_block = f"EVENTS:\n{linearize_events(out['events'])}\n"
    event_lookup = _event_lookup(out["events"])

    out["linearized_events_only"] = events_block

    for condition, sections in _CONDITION_SPEC.items():
        block = out["conditions"][condition]
        relations = block.get("relations") or {}
        parts = [events_block]
        for subkey, header in sections:
            triples = relations.get(subkey) or []
            parts.append(linearize_relations_hybrid(triples, header, event_lookup))
        block["linearized"] = "".join(parts)

    return out


import argparse
import json
import os
from pathlib import Path
from typing import Sequence


def _stream_linearize(in_path: Path, out_path: Path) -> int:
    """Read in_path line by line, apply linearize_row, write to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # new_row = linearize_row(row)  # original: bare-id endpoints, e.g. (e2, BEFORE, e1)
            new_row = linearize_row_hybrid(row)  # hybrid: (e2:left|Departing, BEFORE, e1:arrived|Arriving)
            fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(in_path: Path, inplace: bool) -> int:
    if not inplace:
        raise SystemExit("Only --inplace is supported (writes back to --in).")
    tmp = in_path.with_suffix(in_path.suffix + ".tmp")
    _stream_linearize(in_path, tmp)
    os.replace(tmp, in_path)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Linearize experiment.jsonl per condition (UNI-26).")
    ap.add_argument("--in", dest="in_path", required=True, type=Path, help="experiment.jsonl input")
    ap.add_argument("--inplace", action="store_true", required=True, help="rewrite the input file atomically")
    return ap.parse_args(argv)


def cli(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return main(args.in_path, args.inplace)


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(cli())
