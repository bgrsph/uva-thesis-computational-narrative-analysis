"""Compose temporal_causal_independent from temporal + causal Llama runs.

Reads two per-condition JSONLs produced by `infer_relations.py`, inner-joins
on (wikidata_id, summary_id), and emits a third JSONL whose `condition_block`
holds the union of the temporal + causal relations under `source: "composed"`.

No LLM call, no GPU. Runs locally in seconds. See UNI-65.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main(temporal_path: Path, causal_path: Path, output_path: Path) -> int:
    by_key_t = {(r["wikidata_id"], r["summary_id"]): r for r in _read_jsonl(temporal_path)}
    by_key_c = {(r["wikidata_id"], r["summary_id"]): r for r in _read_jsonl(causal_path)}
    # Only compose for keys where Llama actually ran on BOTH sides. Skip when
    # either side is source='skipped_ctx_overflow' (or any other non-llama
    # source) — the merger will surface those overflow rows under their own
    # condition slots; tcindep is just absent for that summary.
    keys = sorted(
        k for k in (by_key_t.keys() & by_key_c.keys())
        if by_key_t[k]["condition_block"]["source"] == "llama"
        and by_key_c[k]["condition_block"]["source"] == "llama"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for k in keys:
            t = by_key_t[k]
            c = by_key_c[k]
            row = {key: val for key, val in t.items() if key != "condition_block"}
            row["condition_block"] = {
                "source":        "composed",
                "composed_from": ["temporal", "causal"],
                "relations": {
                    "temporal_relations": (t["condition_block"]["relations"] or {}).get("temporal_relations", []),
                    "causal_relations":   (c["condition_block"]["relations"] or {}).get("causal_relations",   []),
                },
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compose temporal_causal_independent from temporal + causal runs.")
    ap.add_argument("--temporal", required=True, type=Path, help="jsonl produced by infer_relations.py for condition=temporal")
    ap.add_argument("--causal",   required=True, type=Path, help="jsonl produced by infer_relations.py for condition=causal")
    ap.add_argument("--output",   required=True, type=Path, help="output jsonl path")
    return ap.parse_args(argv)


def cli(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return main(args.temporal, args.causal, args.output)


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(cli())
