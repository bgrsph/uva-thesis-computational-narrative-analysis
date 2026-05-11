"""Standalone CLI mirror of notebook §4 cells 3–5.

Reuses the pure helpers in models/llama/inference.py end-to-end. The notebook
remains the local-MPS validation path; this script is the cluster-execution
path (invoked by models/llama/infer_relations.sbatch).

See spec: docs/superpowers/specs/2026-05-11-snellius-cluster-execution-design.md
Linear: UNI-66.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

VALID_CONDITIONS = ("temporal", "causal", "temporal_causal_independent", "temporal_causal_joint")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Llama-3-8B relation annotation over BERT+CRF event-tagged jsonl.")
    ap.add_argument("--input",       required=True, type=Path, help="jsonl produced by infer_tma.py")
    ap.add_argument("--output",      required=True, type=Path, help="jsonl to write; one row per processed input row")
    ap.add_argument("--condition",   required=True, choices=VALID_CONDITIONS,
                    help="Prompt condition to run (selects models/llama/prompts/<condition>.yaml)")
    ap.add_argument("--sample-size", type=int, default=None,
                    help="If set, process only the first N input rows (sanity / smoke runs).")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Inference loop lives in run_inference(); split keeps parse_args testable
    # without importing torch/transformers.
    return run_inference(args)


def run_inference(args: argparse.Namespace) -> int:
    raise NotImplementedError("Task 3 fills this in.")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
