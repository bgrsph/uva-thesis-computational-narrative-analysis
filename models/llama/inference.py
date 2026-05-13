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

    Section-mismatch recovery: for multi-section conditions like
    `temporal_causal_independent`, a relation whose label is valid in *another*
    section of this condition's codebook is moved to that section instead of
    rejected. Single-section conditions still raise on label miss. See UNI-74.

    Raises:
        json.JSONDecodeError: output is not valid JSON.
        ValueError: relation label not in *any* of this condition's allow-lists,
            or source/target does not match `^e\\d+$`. (The "is this eID actually
            present in the summary?" check is UNI-60's job, not UNI-12's.)
    """
    parsed = json.loads(out_str)
    sections = CONDITION_KEYS[condition]
    section_allowed = [set(cfg[label_field]) for _, label_field in sections]

    # First pass: route every relation into the section whose codebook accepts its label.
    for cur_idx, (json_key, _) in enumerate(sections):
        for rel in list(parsed[json_key]):   # snapshot — we mutate the live list below
            label = rel["relation"]
            if label in section_allowed[cur_idx]:
                continue
            target_idx = next(
                (i for i, allowed in enumerate(section_allowed)
                 if i != cur_idx and label in allowed),
                None,
            )
            if target_idx is None:
                raise ValueError(f"unknown label: {label}")
            parsed[json_key].remove(rel)
            parsed[sections[target_idx][0]].append(rel)

    # Second pass: now that every relation sits in its correct section, validate eIDs.
    for json_key, _ in sections:
        for rel in parsed[json_key]:
            for end in ("source", "target"):
                if not RE_EID.match(rel[end]):
                    raise ValueError(f"bad eID: {rel[end]}")
    return parsed


def build_chat_messages(cfg: dict, annotated_text: str) -> list[dict]:
    """Assemble the Llama-3 chat-template messages list from a prompt config and the annotated text."""
    return [
        {"role": "system", "content": cfg["system"]},
        {"role": "user",   "content": cfg["user_template"].format(annotated_text=annotated_text)},
    ]


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
