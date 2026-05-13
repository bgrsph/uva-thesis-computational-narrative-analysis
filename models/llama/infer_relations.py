"""Standalone CLI mirror of notebook §4 cells 3–5.

Reuses the pure helpers in models/llama/inference.py end-to-end. The notebook
remains the local-MPS validation path; this script is the cluster-execution
path (invoked by models/llama/infer_relations.sbatch).

See spec: docs/superpowers/specs/2026-05-11-snellius-cluster-execution-design.md
Linear: UNI-66.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from models.llama.inference import (
    build_chat_messages,
    inline_events,
    load_prompt_config,
    parse_and_validate,
)

VALID_CONDITIONS = ("temporal", "causal", "temporal_causal_independent", "temporal_causal_joint")

LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
LLAMA_DO_SAMPLE = False
# Llama-3-8B architectural ctx window; input + output cannot exceed this.
# We let max_new_tokens float per-row as `LLAMA_CTX - n_input` to leave no
# generation budget on the table — see UNI-52 audit.
LLAMA_CTX = 8192


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Llama-3-8B relation annotation over BERT+CRF event-tagged jsonl.")
    ap.add_argument("--input",       required=True, type=Path, help="jsonl produced by infer_tma.py")
    ap.add_argument("--output",      required=True, type=Path, help="jsonl to write; one row per processed input row")
    ap.add_argument("--condition",   required=True, choices=VALID_CONDITIONS,
                    help="Prompt condition to run (selects models/llama/prompts/<condition>.yaml)")
    ap.add_argument("--sample-size", type=int, default=None,
                    help="If set, process only the first N input rows (sanity / smoke runs).")
    return ap.parse_args(argv)


def _build_pipeline(model_id: str):
    """Build the transformers text-generation pipeline. Isolated so tests can monkeypatch it."""
    import transformers  # local import; keeps argparse-only paths free of torch/transformers.
    from dotenv import load_dotenv

    load_dotenv()   # picks up HF_TOKEN from .env if present; harmless if env already exports it.
    return transformers.pipeline(
        "text-generation",
        model=model_id,
        dtype="auto",
        device_map="auto",
    )


def run_inference(args: argparse.Namespace) -> int:
    cfg = load_prompt_config(args.condition)
    pipe = _build_pipeline(LLAMA_MODEL_ID)
    tok = pipe.tokenizer
    terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if args.sample_size is not None and i >= args.sample_size:
                break
            row = json.loads(line)
            annotated = inline_events(row["sentences"], row["events"])
            messages = build_chat_messages(cfg, annotated)

            prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            n_input = len(tok.encode(prompt_str, add_special_tokens=False))

            max_new = LLAMA_CTX - n_input
            if max_new <= 0:
                print(f"row {i}: skipping — input alone exceeds ctx "
                      f"(n_input={n_input} >= {LLAMA_CTX})", flush=True)
                continue

            outputs = pipe(
                messages,
                max_new_tokens=max_new,
                do_sample=LLAMA_DO_SAMPLE,
                eos_token_id=terminators,
                pad_token_id=tok.eos_token_id,
            )
            out_str = outputs[0]["generated_text"][-1]["content"]
            n_output = len(tok.encode(out_str, add_special_tokens=False))
            print(f"row {i}: input_tokens={n_input} output_tokens={n_output}", flush=True)

            try:
                parsed = parse_and_validate(out_str, cfg, args.condition)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"row {i}: {e}", flush=True)
                continue   # spec §3.2 + §5: skip row; UNI-25 owns retry/drop policy.

            fout.write(json.dumps({**row, "relations": parsed}) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_inference(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
