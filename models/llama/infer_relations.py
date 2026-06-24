
# Import necessary libraries and functions
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Sequence

from models.llama.inference import (
    build_chat_messages,
    build_run_row,
    inline_events,
    load_prompt_config,
)

# Define all conditions that require LLM call. Note that indepdently merged temporal+causal does not require LLM call, it's merged from temporal and causal results post-hoc. 
VALID_CONDITIONS = ("temporal", "causal", "temporal_causal_joint")
LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"              # Full official name of the model
LLAMA_DO_SAMPLE = False                                             # For deterministic results
LLAMA_CTX = 8192                                                    # Llama-3-8B architectural context window

# Parse arguments for the python script
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
    """Build the transformers text-generation pipeline."""
    import transformers
    from dotenv import load_dotenv

    load_dotenv()   # picks up HF_TOKEN from .env if present, though we use HF login in this experiment. 
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
    terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]   # stop at end-of-text or Llama-3 end-of-turn

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resume: if the output already holds rows from a prior run that crashed or
    # timed out, skip the inputs already processed and append. 
    # We use the key: (wikidata_id, summary_id).
    # Any trailing partial/corrupt line is truncated so appended rows stay parseable. 
    # Decoding is deterministic, so a skipped row would reproduce exactly. 
    done_keys: set[tuple] = set()
    if args.output.exists():
        valid_bytes = 0
        with args.output.open("rb") as fprev:
            for raw in fprev:
                try:
                    prev = json.loads(raw)
                except json.JSONDecodeError:
                    break   # first corrupt/partial line — truncate from here
                if isinstance(prev, dict):
                    done_keys.add((prev.get("wikidata_id"), prev.get("summary_id")))
                valid_bytes += len(raw)
        with args.output.open("r+b") as fprev:
            fprev.truncate(valid_bytes)   # drop any partial trailing write
    if done_keys:
        print(f"resume: {len(done_keys)} rows already in {args.output.name}; skipping them", flush=True)

    mode = "a" if done_keys else "w"   # append when resuming a prior run, else write a fresh file
    with args.input.open("r", encoding="utf-8") as fin, args.output.open(mode, encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if args.sample_size is not None and i >= args.sample_size:
                break
            row = json.loads(line)
            if (row.get("wikidata_id"), row.get("summary_id")) in done_keys:
                continue   # already processed in a prior run — skip
            annotated = inline_events(row["sentences"], row["events"])
            messages = build_chat_messages(cfg, annotated)
            prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            n_input = len(tok.encode(prompt_str, add_special_tokens=False))
            max_new = LLAMA_CTX - n_input   # tokens left in the context window for the response
            
            if max_new <= 0:
                print(f"row {i}: skipping — input alone exceeds ctx "
                      f"(n_input={n_input} >= {LLAMA_CTX})", flush=True)
                overflow_row = {
                    **row,
                    "condition_block": {
                        "source":          "skipped_ctx_overflow",
                        "model_id":        LLAMA_MODEL_ID,
                        "prompt_template": f"models/llama/prompts/{args.condition}.yaml",
                        "prompt_rendered": prompt_str,
                        "response_raw":    None,
                        "response_parsed": None,
                        "parse_error":     None,
                        "relations":       None,
                        "input_tokens":    n_input,
                        "output_tokens":   None,
                        "max_new_tokens":  None,
                        "hit_ctx_cap":     None,
                    },
                }
                fout.write(json.dumps(overflow_row) + "\n")
                continue

            # Run the model with greedy, deterministic decoding, stopping at the terminators.
            outputs = pipe(
                messages,
                max_new_tokens=max_new,
                do_sample=LLAMA_DO_SAMPLE,
                eos_token_id=terminators,
                pad_token_id=tok.eos_token_id,
            )
            # Take the assistant's reply text and count its output tokens.
            out_str = outputs[0]["generated_text"][-1]["content"]
            n_output = len(tok.encode(out_str, add_special_tokens=False))
            print(f"row {i}: input_tokens={n_input} output_tokens={n_output}", flush=True)

            # Parse and validate the reply into the output row (relations, parse status, token counts, metadata).
            out_row = build_run_row(
                input_row=row,
                condition=args.condition,
                model_id=LLAMA_MODEL_ID,
                prompt_rendered=prompt_str,
                response_raw=out_str,
                input_tokens=n_input,
                output_tokens=n_output,
                max_new_tokens=max_new,
                cfg=cfg,
            )
            # Report any parse failure, then append the row to the output file.
            if out_row["condition_block"]["parse_error"]:
                print(f"row {i}: {out_row['condition_block']['parse_error']}", flush=True)
            fout.write(json.dumps(out_row) + "\n")
    return 0

# Define the run
def main(argv: Sequence[str] | None = None) -> int:
    return run_inference(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
