"""One-time migration for UNI-88: relocate the current loose experiment
artifacts into a single per-experiment folder. Non-destructive (moves, no
renames). Dry-run by default; pass --apply to execute.

Run from the repo root:
    venv/bin/python scripts/migrate_experiment_layout.py          # preview
    venv/bin/python scripts/migrate_experiment_layout.py --apply  # do it
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.experiment_paths import new_experiment_name

INTER = ROOT / "data" / "intermediate"
RESULTS = ROOT / "data" / "results"
SPLIT = "test"


def main(apply: bool) -> int:
    subset = INTER / f"tma_subset.{SPLIT}.jsonl"
    experiment = RESULTS / "experiment.jsonl"
    if not subset.exists() or not experiment.exists():
        print(f"Nothing to migrate: missing {subset} or {experiment}.")
        return 0

    n_rows = sum(1 for _ in subset.open(encoding="utf-8"))
    ts = datetime.fromtimestamp(experiment.stat().st_mtime).strftime("%Y%m%d_%H%M")
    name = new_experiment_name(SPLIT, n_rows, ts=ts)   # same format as the notebook
    dest = ROOT / "data" / "experiments" / name

    # (source, destination-relative-to-dest). None dest-dir => place at folder root.
    moves: list[tuple[Path, Path]] = [
        (experiment,                               dest / "experiment.jsonl"),
        (RESULTS / "experiment.jsonl.bak",         dest / "experiment.jsonl.bak"),
        (INTER / "similarities.npz",               dest / "similarities.npz"),
        (INTER / "baselines" / "vocab.json",       dest / "baselines" / "vocab.json"),
        (subset,                                   dest / f"tma_subset.{SPLIT}.jsonl"),
        (INTER / f"tma_subset_events_full.{SPLIT}.jsonl",
                                                   dest / f"tma_subset_events_full.{SPLIT}.jsonl"),
    ]
    for sub in ("embeddings", "llama_runs"):
        src_dir = INTER / sub
        if src_dir.is_dir():
            for f in sorted(src_dir.glob("*.jsonl")):
                moves.append((f, dest / sub / f.name))

    print(f"Target folder: {dest.relative_to(ROOT)}")
    planned = [(s, d) for s, d in moves if s.exists()]
    for s, d in planned:
        print(f"  {'MOVE' if apply else 'plan'}: {s.relative_to(ROOT)}  ->  {d.relative_to(ROOT)}")
    skipped = [s for s, _ in moves if not s.exists()]
    for s in skipped:
        print(f"  skip (absent): {s.relative_to(ROOT)}")

    if not apply:
        print("\nDry run. Re-run with --apply to perform the moves.")
        return 0

    for s, d in planned:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
    print(f"\nMigrated {len(planned)} artifacts into {dest.relative_to(ROOT)}")
    print(f"Set in the notebooks:  EXPERIMENT_NAME = \"{name}\"")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="perform the moves (default: dry run)")
    sys.exit(main(p.parse_args().apply))
