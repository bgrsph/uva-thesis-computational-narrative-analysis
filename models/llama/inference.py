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


def parse_and_validate(out_str: str, cfg: dict, condition: str) -> dict:
    """Parse the model's raw output and validate label / eID shape.

    Returns the cleaned dict of per-section relation lists. Relations that are
    malformed in shape (not an object, missing `relation`/`source`/`target`,
    unknown label, or malformed eID) are silently dropped; the verbatim Llama
    output is preserved in `condition_block.response_raw` so anyone wanting
    to recover them can re-parse from there. UNI-65 Option A.

    Section-mismatch recovery: for multi-section conditions, a relation whose
    label is valid in *another* section of this condition's codebook is moved
    to that section. (Single-section is what we have today; the reroute is
    dead code under current `CONDITION_KEYS` but kept for a future multi-
    section condition.)

    Raises:
        json.JSONDecodeError: output is not valid JSON (the only hard-fail
            mode — everything past JSON parse is recoverable).
    """
    parsed = json.loads(out_str)
    sections = CONDITION_KEYS[condition]
    section_allowed = [set(cfg[label_field]) for _, label_field in sections]
    out: dict[str, list[dict]] = {json_key: [] for json_key, _ in sections}

    for cur_idx, (json_key, _) in enumerate(sections):
        for rel in parsed.get(json_key, []):
            if not isinstance(rel, dict):
                continue   # malformed entry (not an object) — silent drop
            label = rel.get("relation")
            target_idx = cur_idx if label in section_allowed[cur_idx] else next(
                (i for i, allowed in enumerate(section_allowed)
                 if i != cur_idx and label in allowed),
                None,
            )
            if target_idx is None:
                continue   # unknown/missing label — silent drop; response_raw is the audit trail
            src, tgt = rel.get("source"), rel.get("target")
            if not (isinstance(src, str) and isinstance(tgt, str)
                    and RE_EID.match(src) and RE_EID.match(tgt)):
                continue   # missing or malformed eID — same
            out[sections[target_idx][0]].append(rel)

    return out


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
      `response_parsed` / `relations` as None. `response_raw` still verbatim.
    - Per-relation validation issues (bad labels, bad eIDs): silently dropped
      in `parse_and_validate`; the valid relations survive. `parse_error` is
      None — `response_raw` is the audit trail for what got dropped.
    """
    try:
        parsed = parse_and_validate(response_raw, cfg, condition)
        parse_error = None
    except json.JSONDecodeError as e:
        parsed = None
        parse_error = f"{type(e).__name__}: {e}"

    return {
        **input_row,
        "condition_block": {
            "source":          "llama",
            "model_id":        model_id,
            "prompt_template": f"models/llama/prompts/{condition}.yaml",
            "prompt_rendered": prompt_rendered,
            "response_raw":    response_raw,
            "response_parsed": parsed,
            "parse_error":     parse_error,
            "relations":       parsed,
            "input_tokens":    input_tokens,
            "output_tokens":   output_tokens,
            "max_new_tokens":  max_new_tokens,
            "hit_ctx_cap":     output_tokens == max_new_tokens,
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
