"""Pure logic for Llama-3 relation annotation. No torch / transformers imports at module level."""
import json
import re
from pathlib import Path

import yaml

RE_EID = re.compile(r"^e\d+$")

CONDITION_KEYS: dict[str, list[tuple[str, str]]] = {
    "temporal":              [("temporal_relations", "allowed_labels")],
    "causal":                [("causal_relations",   "allowed_labels")],
    "temporal_causal_joint": [("joint_relations",    "allowed_labels")],
}


def load_prompt_config(condition: str) -> dict:
    """Load the prompt YAML for the given condition.

    The YAMLs at `models/llama/prompts/*.yaml` are frozen per UNI-13;
    a missing field surfaces as the caller's KeyError.
    """
    path = Path(__file__).parent / "prompts" / f"{condition}.yaml"
    return yaml.safe_load(path.read_text())


def parse_and_validate(out_str: str, cfg: dict, condition: str) -> tuple[dict, list[dict]]:
    """Parse the model's raw output and validate label / eID shape.

    Returns ``(parsed, rejected)`` — a dict of cleaned per-section relation lists
    and a list of rejected entries. **Bad labels and bad eIDs no longer raise;**
    they are dropped from `parsed` and appended to `rejected` with a reason.
    This was UNI-65's "Option A" recovery: previously a single bad relation
    would null the whole row.

    Section-mismatch recovery: for multi-section conditions, a relation whose
    label is valid in *another* section of this condition's codebook is moved
    to that section instead of being rejected. (Single-section is what we have
    today; the reroute is dead code under current `CONDITION_KEYS` but kept for
    a future multi-section condition.)

    Each rejection record has shape::

        {"section": "<json_key>", "relation": <orig rel dict>, "reason": "<str>"}

    Raises:
        json.JSONDecodeError: output is not valid JSON (the only remaining
            hard-fail mode — everything past JSON parse is recoverable).
    """
    parsed = json.loads(out_str)
    sections = CONDITION_KEYS[condition]
    section_allowed = [set(cfg[label_field]) for _, label_field in sections]
    rejected: list[dict] = []
    out: dict[str, list[dict]] = {json_key: [] for json_key, _ in sections}

    # Pass 1: route by label. Unknown-label relations land in `rejected`.
    for cur_idx, (json_key, _) in enumerate(sections):
        for rel in parsed.get(json_key, []):
            label = rel["relation"]
            if label in section_allowed[cur_idx]:
                out[json_key].append(rel)
                continue
            target_idx = next(
                (i for i, allowed in enumerate(section_allowed)
                 if i != cur_idx and label in allowed),
                None,
            )
            if target_idx is None:
                rejected.append({"section": json_key, "relation": rel,
                                 "reason": f"unknown label: {label}"})
                continue
            out[sections[target_idx][0]].append(rel)

    # Pass 2: validate eIDs. Bad-eID relations move from `out` to `rejected`.
    for json_key, _ in sections:
        kept: list[dict] = []
        for rel in out[json_key]:
            bad_eid = next(
                (rel[end] for end in ("source", "target") if not RE_EID.match(rel[end])),
                None,
            )
            if bad_eid is not None:
                rejected.append({"section": json_key, "relation": rel,
                                 "reason": f"bad eID: {bad_eid}"})
                continue
            kept.append(rel)
        out[json_key] = kept

    return out, rejected


def build_chat_messages(cfg: dict, annotated_text: str) -> list[dict]:
    """Assemble the Llama-3 chat-template messages list from a prompt config and the annotated text."""
    return [
        {"role": "system", "content": cfg["system"]},
        {"role": "user",   "content": cfg["user_template"].format(annotated_text=annotated_text)},
    ]


def build_run_row(
    *,
    input_row: dict,
    condition: str,
    model_id: str,
    prompt_rendered: str,
    response_raw: str,
    input_tokens: int,
    output_tokens: int,
    max_new_tokens: int,
    cfg: dict,
) -> dict:
    """Compose the per-condition row written by infer_relations.py.

    Returns a row whether or not parsing succeeded.

    - JSON-parse failure (the only hard-fail): populates `parse_error`, leaves
      `response_parsed` / `relations` as None, and `rejected_relations` as [].
    - Per-relation validation issues (bad labels, bad eIDs): the offending
      relations land in `rejected_relations`; the valid ones are still in
      `relations`. `parse_error` is None in this case. This is UNI-65's
      Option A recovery — previously, a single bad relation nulled the row.
    """
    try:
        parsed, rejected = parse_and_validate(response_raw, cfg, condition)
        parse_error = None
    except json.JSONDecodeError as e:
        parsed = None
        rejected = []
        parse_error = f"{type(e).__name__}: {e}"

    return {
        **input_row,
        "condition_block": {
            "source":             "llama",
            "model_id":           model_id,
            "prompt_template":    f"models/llama/prompts/{condition}.yaml",
            "prompt_rendered":    prompt_rendered,
            "response_raw":       response_raw,
            "response_parsed":    parsed,
            "parse_error":        parse_error,
            "relations":          parsed,
            "rejected_relations": rejected,
            "input_tokens":       input_tokens,
            "output_tokens":      output_tokens,
            "max_new_tokens":     max_new_tokens,
            "hit_ctx_cap":        output_tokens == max_new_tokens,
        },
    }


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
