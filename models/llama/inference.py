"""Pure logic for Llama-3 relation annotation. No torch / transformers imports at module level."""
import json
import re
from pathlib import Path

import yaml

RE_EID = re.compile(r"^e\d+$")

CONDITION_KEYS: dict[str, list[tuple[str, str]]] = {
    "temporal":                    [("temporal_relations", "allowed_labels")],
    "causal":                      [("causal_relations",   "allowed_labels")],
    "temporal_causal_independent": [("temporal_relations", "allowed_temporal_labels"),
                                    ("causal_relations",   "allowed_causal_labels")],
    "temporal_causal_joint":       [("joint_relations",    "allowed_labels")],
}


def load_prompt_config(condition: str) -> dict:
    """Load the prompt YAML for the given condition.

    The YAMLs at `models/llama/prompts/*.yaml` are frozen per UNI-13;
    a missing field surfaces as the caller's KeyError.
    """
    path = Path(__file__).parent / "prompts" / f"{condition}.yaml"
    return yaml.safe_load(path.read_text())


def parse_and_validate(out_str: str, cfg: dict, condition: str) -> dict:
    """Parse the model's raw output and validate label / eID shape.

    Raises:
        json.JSONDecodeError: output is not valid JSON.
        ValueError: relation label not in the YAML allow-list, or source/target
            does not match `^e\\d+$`. (The "is this eID actually present in the
            summary?" check is UNI-60's job, not UNI-12's.)
    """
    parsed = json.loads(out_str)
    for json_key, label_field in CONDITION_KEYS[condition]:
        allowed = set(cfg[label_field])
        for rel in parsed[json_key]:
            if rel["relation"] not in allowed:
                raise ValueError(f"unknown label: {rel['relation']}")
            for end in ("source", "target"):
                if not RE_EID.match(rel[end]):
                    raise ValueError(f"bad eID: {rel[end]}")
    return parsed


def inline_events(sentences: list[str], events: list[dict]) -> str:
    """Splice `[event_id|trigger|event_type]` markers into each sentence at the event spans.

    Uses the `event_id` already assigned by the upstream BERT+CRF stage
    (`models/bert_crf/infer_tma.py` assigns e1, e2, ... in document order).
    """
    by_sent: dict[int, list[dict]] = {}
    for ev in events:
        by_sent.setdefault(ev["sent_id"], []).append(ev)

    out: list[str] = []
    for sent_id, sent in enumerate(sentences):
        evs = sorted(by_sent.get(sent_id, []), key=lambda e: e["start"])
        pieces: list[str] = []
        cursor = 0
        for ev in evs:
            pieces.append(sent[cursor:ev["start"]])
            pieces.append(f"[{ev['event_id']}|{ev['trigger']}|{ev['event_type']}]")
            cursor = ev["end"]
        pieces.append(sent[cursor:])
        out.append("".join(pieces))
    return " ".join(out)
