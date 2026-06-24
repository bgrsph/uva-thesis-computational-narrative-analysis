from datetime import datetime

# Build new expeirment name based on meta data and time-stamping
def new_experiment_name(split: str, n_rows: int, ts: str | None = None) -> str:
    """experiment_<split>_<n_rows>_<YYYYMMDD_HHMM>.

    ts overrides the timestamp (used by the migration script to stamp an existing
    run from its mtime); defaults to now.
    """
    return f"experiment_{split}_{n_rows}_{ts or datetime.now().strftime('%Y%m%d_%H%M')}"
