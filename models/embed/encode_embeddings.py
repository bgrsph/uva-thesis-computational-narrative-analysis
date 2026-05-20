"""Streaming CLI: read experiment.jsonl, write per-row embeddings JSONL for one encoder.

Mirrors models/llama/infer_relations.py shape. See UNI-14 design."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from models.embed.encoders import ENCODER_REGISTRY, encode

# Pin the iteration order so downstream code can rely on it.
CONDITION_KEYS: tuple[str, ...] = (
    "raw_text",
    "events_only",
    "temporal",
    "causal",
    "temporal_causal_independent",
    "temporal_causal_joint",
)


def _gather_inputs(row: dict) -> list[str]:
    """Return the six input strings for one row, in CONDITION_KEYS order."""
    return [
        row["text"],
        row["linearized_events_only"],
        row["conditions"]["temporal"]["linearized"],
        row["conditions"]["causal"]["linearized"],
        row["conditions"]["temporal_causal_independent"]["linearized"],
        row["conditions"]["temporal_causal_joint"]["linearized"],
    ]


def _build_model(model_id: str):
    """Build the SentenceTransformer. Isolated so tests can monkeypatch."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id)


def run(args: argparse.Namespace) -> int:
    model_id, task = ENCODER_REGISTRY[args.encoder]
    model = _build_model(model_id)
    dim = int(model.get_embedding_dimension())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as fin, \
         args.output.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts = _gather_inputs(row)
            vecs = encode(model, texts, task, batch_size=args.batch_size)
            out_row = {
                "wikidata_id": row["wikidata_id"],
                "summary_id":  row["summary_id"],
                "encoder_key": args.encoder,
                "model_id":    model_id,
                "task":        task,
                "dim":         dim,
                "vectors":     {k: vecs[j].tolist() for j, k in enumerate(CONDITION_KEYS)},
            }
            fout.write(json.dumps(out_row) + "\n")
            print(f"row {i}: {row['wikidata_id']}/{row['summary_id']}", flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Encode experiment.jsonl rows with one HuggingFace encoder (UNI-14).")
    ap.add_argument("--encoder",    required=True, choices=list(ENCODER_REGISTRY.keys()))
    ap.add_argument("--input",      required=True, type=Path, help="data/results/experiment.jsonl")
    ap.add_argument("--output",     required=True, type=Path,
                    help="data/intermediate/embeddings/<encoder>.jsonl")
    ap.add_argument("--batch-size", type=int, default=8)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
