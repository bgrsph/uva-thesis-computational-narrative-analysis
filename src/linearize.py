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
            new_row = linearize_row(row)
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
