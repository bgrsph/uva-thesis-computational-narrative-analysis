"""Per-experiment folder naming (UNI-88).

One source of truth for the experiment folder name, so every artifact of a run
lands under `data/experiments/<name>/`. A fresh timestamp is minted on each call,
so re-running subsetting never clobbers a prior run.
"""
from datetime import datetime


def new_experiment_name(split: str, n_rows: int, ts: str | None = None) -> str:
    """experiment_<split>_<n_rows>_<YYYYMMDD_HHMM>.

    ts overrides the timestamp (used by the migration script to stamp an existing
    run from its mtime); defaults to now.
    """
    return f"experiment_{split}_{n_rows}_{ts or datetime.now().strftime('%Y%m%d_%H%M')}"
