# Llama-3 zero-shot relation-annotation prompts (UNI-13)

This directory holds the four zero-shot prompt templates used to annotate
temporal and causal relations between BERT+CRF-extracted events in
*Tell Me Again!* summaries. One YAML file per experimental condition; the
inference script (UNI-24) loads the file matching the requested
`--condition` and feeds it into Llama-3-8B-Instruct.



## Files

| File | Condition | Output JSON top-level key(s) | Allowed-label count |
|---|---|---|---|
| `temporal.yaml` | e+t | `temporal_relations` | 4 |
| `causal.yaml` | e+c | `causal_relations` | 4 |
| `temporal_causal_independent.yaml` | e+t+c (independent) | `temporal_relations` + `causal_relations` | 4 + 4 |
| `temporal_causal_joint.yaml` | e+joint | `joint_relations` | 13 (9 fused + 4 fallbacks) |

> **Pending decision (UNI-62):** the `temporal_causal_independent.yaml`
> prompt may be dropped in favour of deriving the e+t+c condition by union
> of the e+t and e+c outputs. Decision is gated on the UNI-24 pilot.

## YAML schema

Each file is a self-contained config with these fields:

| Field | Type | Description |
|---|---|---|
| `condition` | string | One of: `temporal`, `causal`, `temporal_causal_independent`, `temporal_causal_joint`. Matches the filename. |
| `system` | string (multiline) | Full Llama-3 system message — role description, codebook, formatting constraints. Identical across all summaries in this condition. |
| `user_template` | string (multiline) | Per-summary user message with a single `{annotated_text}` placeholder. Identical across all four files. |
| `allowed_labels` | list of strings | Allow-list of valid relation labels for parser-side validation. Used by all conditions *except* `temporal_causal_independent`. |
| `allowed_temporal_labels` | list of strings | (`temporal_causal_independent` only) allow-list for the `temporal_relations` array. |
| `allowed_causal_labels` | list of strings | (`temporal_causal_independent` only) allow-list for the `causal_relations` array. |

Decoding parameters (`temperature`, `max_new_tokens`, etc.) and the model ID
are **not** in the YAML — they are condition-invariant and live as constants
in `pipeline.ipynb §4` (`LLAMA_MODEL_ID`, `LLAMA_TEMPERATURE`,
`LLAMA_MAX_NEW_TOKENS`, `LLAMA_FALLBACK_MAX_NEW_TOKENS`,
`LLAMA_DO_SAMPLE`).

## Loader contract

```python
import yaml, json, re

# Load the prompt config for the chosen condition
cfg = yaml.safe_load(open(f"models/llama/prompts/{cond}.yaml"))

# Build the chat-template messages
messages = [
    {"role": "system", "content": cfg["system"]},
    {"role": "user",   "content": cfg["user_template"].format(annotated_text=row["annotated_text"])},
]
prompt_str = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# ... model.generate() with LLAMA_* constants from pipeline.ipynb §4 ...

# Parser-side validation against the YAML's allow-list(s)
RE_EID = re.compile(r"^e\d+$")

CONDITION_KEYS = {
    "temporal":                     [("temporal_relations", "allowed_labels")],
    "causal":                       [("causal_relations",   "allowed_labels")],
    "temporal_causal_independent":  [("temporal_relations", "allowed_temporal_labels"),
                                     ("causal_relations",   "allowed_causal_labels")],
    "temporal_causal_joint":        [("joint_relations",    "allowed_labels")],
}

parsed = json.loads(out_str)
for json_key, label_field in CONDITION_KEYS[cond]:
    allowed = set(cfg[label_field])
    for rel in parsed[json_key]:
        assert rel["relation"] in allowed
        assert RE_EID.match(rel["source"]) and RE_EID.match(rel["target"])
```

The full validator (with retry/drop policies for malformed JSON, unknown
labels, and hallucinated event IDs) is specified in §8 of the spec.

## Updating these files

These prompts are **frozen for the full corpus run**. Any changes to the
codebook or formatting after UNI-24's pilot complete are out of scope for
UNI-13 and require a follow-up issue. The thesis methodology (§3.2.3 +
Appendix C) must stay synchronised with this directory — that
synchronisation is tracked in **UNI-61**.

## Cross-references

- **UNI-13** (parent) — Prompt design.
- **UNI-24** — Pilot annotation batch + codebook calibration.
- **UNI-26** — Narrative linearization for the embedder.
- **UNI-52** — Llama-3 context-budget audit.
- **UNI-60** — LLM event-ID hallucination rate measurement.
- **UNI-61** — Thesis §3.2.3 + Appendix C update.
- **UNI-62** — Decide whether to drop `temporal_causal_independent.yaml`.
