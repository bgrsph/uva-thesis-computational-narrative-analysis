# Import necessary libraries
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Sequence

# Read the JSON file via inner _function
def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main(experiment_path: Path, encoder_paths: list[Path], inplace: bool) -> int:
    
    # We need this safety swtich here, because we want one experiment.jsonl file to be used to communicate with cluster
    if not inplace:
        raise SystemExit("Only --inplace is supported (writes back to --in).")

    # Read all the rows from the experiment.jsonl, then put them into a dictiionary with identification
    rows: list[dict] = _read_jsonl(experiment_path)
    by_key: dict[tuple, dict] = {(r["wikidata_id"], r["summary_id"]): r for r in rows}

    # For each encoded path, extract the information
    for enc_path in encoder_paths:
        for er in _read_jsonl(enc_path):
            key = (er["wikidata_id"], er["summary_id"])
            target = by_key.get(key)
            if target is None:
                continue   # encoder produced a row we don't have in experiment.jsonl: inner-join skip
            target.setdefault("embeddings", {})[er["encoder_key"]] = {
                "model_id": er["model_id"],
                "task":     er["task"],
                "dim":      er["dim"],
                "vectors":  er["vectors"],
            }
    
    # Since we write experiment.jsonl inline, safety measurement is creating a temporary file incase something goes wrong in the middle of run
    tmp = experiment_path.with_suffix(experiment_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fout:
        for r in rows:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Replace the temporay file with actual experiment.jsonl
    os.replace(tmp, experiment_path)
    return 0

# Parse arguments for the python script
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge per-encoder embeddings into experiment.jsonl.")
    ap.add_argument("--in",       dest="experiment_path", required=True, type=Path)
    ap.add_argument("--encoders", required=True, type=Path, nargs="+")
    ap.add_argument("--inplace",  action="store_true", required=True)
    return ap.parse_args(argv)

# Define the run
def cli(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return main(args.experiment_path, args.encoders, args.inplace)


if __name__ == "__main__":   
    raise SystemExit(cli())
