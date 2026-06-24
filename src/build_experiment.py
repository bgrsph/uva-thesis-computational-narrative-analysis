# Import necessary libraries
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Sequence

# Read the JSON file via inner _function
def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

def main(
    temporal_path: Path,
    causal_path: Path,
    tcjoint_path: Path,
    tcindep_path: Path,
    output_path: Path,
) -> int:
    sources = {
        "temporal":                    temporal_path,
        "causal":                      causal_path,
        "temporal_causal_joint":       tcjoint_path,
        "temporal_causal_independent": tcindep_path,
    }
    rows: dict[tuple, dict] = {}

    # For each relational condition, find the data and insert into rows
    for condition, path in sources.items():
        for r in _read_jsonl(path):
            key = (r["wikidata_id"], r["summary_id"])
            if key not in rows:
                rows[key] = {k: v for k, v in r.items() if k != "condition_block"}
                rows[key]["conditions"] = {}
                rows[key]["embeddings"] = {}
                rows[key]["baselines"]  = {}
            rows[key]["conditions"][condition] = r["condition_block"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for key in sorted(rows):
            fout.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
    return 0

# Parse arguments for the python script
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge per-condition Llama runs into a single experiment.jsonl.")
    ap.add_argument("--temporal", required=True, type=Path)
    ap.add_argument("--causal",   required=True, type=Path)
    ap.add_argument("--tcjoint",  required=True, type=Path, help="temporal_causal_joint.jsonl")
    ap.add_argument("--tcindep",  required=True, type=Path, help="temporal_causal_independent.jsonl (composed)")
    ap.add_argument("--output",   required=True, type=Path, help="experiment.jsonl output path")
    return ap.parse_args(argv)

# Define the run
def cli(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return main(args.temporal, args.causal, args.tcjoint, args.tcindep, args.output)


if __name__ == "__main__": 
    raise SystemExit(cli())
