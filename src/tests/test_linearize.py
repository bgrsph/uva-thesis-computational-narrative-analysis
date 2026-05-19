"""Unit tests for src/linearize.py — the UNI-26 canonical linearization templater."""
import json
import re

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


def test_relation_sort_by_eid():
    triples = [
        {"source": "e17", "target": "e16", "relation": "CAUSE"},
        {"source": "e12", "target": "e11", "relation": "CAUSE"},
    ]
    out = linearize.linearize_relations(triples, "CAUSAL")
    assert out == "CAUSAL:\n(e12, CAUSE, e11)\n(e17, CAUSE, e16)\n"


def test_empty_relations_keeps_header():
    out = linearize.linearize_relations([], "CAUSAL")
    assert out == "CAUSAL:\n"


def _row_fixture():
    """Minimal row matching the experiment.jsonl schema for a 3-event story."""
    return {
        "wikidata_id": "1000000",
        "summary_id":  "en",
        "lang":        "en",
        "text":        "Alice arrived. Bob left. Carol stayed.",
        "events": [
            _event("e1", 0,  0, "arrived",  "Arriving"),
            _event("e2", 1,  4, "left",     "Departing"),
            _event("e3", 2,  6, "stayed",   "Stay"),
        ],
        "conditions": {
            "temporal": {
                "relations":   {"temporal_relations": [
                    {"source": "e2", "target": "e1", "relation": "BEFORE"},
                ]},
                "parse_error": None,
            },
            "causal": {
                "relations":   {"causal_relations": [
                    {"source": "e1", "target": "e2", "relation": "CAUSE"},
                    {"source": "e2", "target": "e3", "relation": "ENABLE"},
                ]},
                "parse_error": None,
            },
            "temporal_causal_joint": {
                "relations":   {"joint_relations": [
                    {"source": "e1", "target": "e2", "relation": "CAUSE_BEFORE"},
                ]},
                "parse_error": None,
            },
            "temporal_causal_independent": {
                "relations":   {
                    "temporal_relations": [
                        {"source": "e2", "target": "e1", "relation": "BEFORE"},
                    ],
                    "causal_relations": [
                        {"source": "e1", "target": "e2", "relation": "CAUSE"},
                    ],
                },
            },
        },
    }


def test_linearize_row_populates_five_keys():
    out = linearize.linearize_row(_row_fixture())
    assert out["linearized_events_only"] == "e1|arrived|Arriving e2|left|Departing e3|stayed|Stay"
    assert out["conditions"]["temporal"]["linearized"] == (
        "e1|arrived|Arriving e2|left|Departing e3|stayed|Stay\n"
        "TEMPORAL:\n"
        "(e2, BEFORE, e1)\n"
    )
    assert out["conditions"]["causal"]["linearized"] == (
        "e1|arrived|Arriving e2|left|Departing e3|stayed|Stay\n"
        "CAUSAL:\n"
        "(e1, CAUSE, e2)\n"
        "(e2, ENABLE, e3)\n"
    )
    assert out["conditions"]["temporal_causal_joint"]["linearized"] == (
        "e1|arrived|Arriving e2|left|Departing e3|stayed|Stay\n"
        "TEMPEROCAUSAL:\n"
        "(e1, CAUSE_BEFORE, e2)\n"
    )


def test_independent_section_order():
    out = linearize.linearize_row(_row_fixture())
    indep = out["conditions"]["temporal_causal_independent"]["linearized"]
    assert indep == (
        "e1|arrived|Arriving e2|left|Departing e3|stayed|Stay\n"
        "TEMPORAL:\n"
        "(e2, BEFORE, e1)\n"
        "CAUSAL:\n"
        "(e1, CAUSE, e2)\n"
    )
    # Belt-and-braces: TEMPORAL header appears before CAUSAL header.
    assert indep.index("TEMPORAL:") < indep.index("CAUSAL:")


def test_parse_error_treated_as_empty_relations():
    row = _row_fixture()
    row["conditions"]["temporal"]["parse_error"] = "JSONDecodeError: ..."
    row["conditions"]["temporal"]["relations"]   = None
    out = linearize.linearize_row(row)
    assert out["conditions"]["temporal"]["linearized"] == (
        "e1|arrived|Arriving e2|left|Departing e3|stayed|Stay\n"
        "TEMPORAL:\n"
    )


def test_does_not_mutate_input():
    row = _row_fixture()
    before = json.dumps(row, sort_keys=True)
    linearize.linearize_row(row)
    after = json.dumps(row, sort_keys=True)
    assert before == after


def test_determinism_round_trip():
    row = _row_fixture()
    out1 = linearize.linearize_row(row)

    # Shuffle the events and relations to challenge the sort.
    row2 = _row_fixture()
    row2["events"] = list(reversed(row2["events"]))
    row2["conditions"]["causal"]["relations"]["causal_relations"] = list(
        reversed(row2["conditions"]["causal"]["relations"]["causal_relations"])
    )
    row2["conditions"]["temporal_causal_independent"]["relations"]["temporal_relations"] = list(
        reversed(row2["conditions"]["temporal_causal_independent"]["relations"]["temporal_relations"])
    )
    out2 = linearize.linearize_row(row2)

    keys = (
        "linearized_events_only",
        ("conditions", "temporal", "linearized"),
        ("conditions", "causal", "linearized"),
        ("conditions", "temporal_causal_joint", "linearized"),
        ("conditions", "temporal_causal_independent", "linearized"),
    )
    for k in keys:
        if isinstance(k, str):
            assert out1[k] == out2[k], f"determinism broken at {k}"
        else:
            a, b = out1, out2
            for part in k:
                a, b = a[part], b[part]
            assert a == b, f"determinism broken at {k}"


from pathlib import Path


def test_cli_inplace_rewrites_file(tmp_path):
    row = _row_fixture()
    path = tmp_path / "experiment.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rc = linearize.cli(["--in", str(path), "--inplace"])
    assert rc == 0

    rewritten = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rewritten) == 1
    r = rewritten[0]
    assert r["linearized_events_only"].startswith("e1|arrived|Arriving")
    assert "TEMPORAL:" in r["conditions"]["temporal"]["linearized"]
    assert "CAUSAL:"  in r["conditions"]["causal"]["linearized"]
    assert "TEMPEROCAUSAL:" in r["conditions"]["temporal_causal_joint"]["linearized"]
    # No stray .tmp left behind.
    assert not (tmp_path / "experiment.jsonl.tmp").exists()


def test_real_row_smoke():
    """Reads the first row of data/results/experiment.jsonl and verifies all
    5 linearized keys are present and shaped correctly."""
    path = Path("data/results/experiment.jsonl")
    if not path.exists():
        pytest.skip("data/results/experiment.jsonl not present in this checkout")
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    out = linearize.linearize_row(row)

    n_events = len(row["events"])
    events_line = out["linearized_events_only"]
    assert events_line, "events line is empty"
    # Triggers may contain spaces (e.g., "sets off"), so count eN| boundaries
    # rather than splitting on space.
    units = re.findall(r"(?:^| )(e\d+)\|", events_line)
    assert len(units) == n_events, (
        f"expected {n_events} event units, got {len(units)}"
    )

    for cond in ("temporal", "causal", "temporal_causal_independent", "temporal_causal_joint"):
        s = out["conditions"][cond]["linearized"]
        assert s.startswith(events_line + "\n"), f"{cond}: events line not a prefix"

    assert "TEMPORAL:"      in out["conditions"]["temporal"]["linearized"]
    assert "CAUSAL:"        in out["conditions"]["causal"]["linearized"]
    assert "TEMPORAL:"      in out["conditions"]["temporal_causal_independent"]["linearized"]
    assert "CAUSAL:"        in out["conditions"]["temporal_causal_independent"]["linearized"]
    assert "TEMPEROCAUSAL:" in out["conditions"]["temporal_causal_joint"]["linearized"]
