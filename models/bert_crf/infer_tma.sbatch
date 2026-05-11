#!/usr/bin/env bash
#SBATCH --job-name=bertcrf-infer-tma
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# CPU-partition wrapper around models/bert_crf/infer_tma.py.
# Partition is NOT set in the header (variable expansion does not work inside
# #SBATCH lines on most clusters). Submitter must pass --partition= on the
# sbatch command line:
#
#   sbatch --partition="$SLURM_PARTITION_CPU" models/bert_crf/infer_tma.sbatch \
#       --input      data/intermediate/tma_subset.jsonl \
#       --output     data/intermediate/tma_subset_events.jsonl \
#       --checkpoint data/intermediate/models/bert_crf \
#       --sample-size 10
#
# All CLI args are forwarded to infer_tma.py. See UNI-66 spec §3.1, §6, §7.1.

set -euo pipefail

: "${SLURM_PARTITION_CPU:?set SLURM_PARTITION_CPU before sbatch (e.g. export SLURM_PARTITION_CPU=rome)}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# shellcheck disable=SC1091
. "$REPO_ROOT/.venv-maven-train/bin/activate"

echo "[infer_tma.sbatch] job=${SLURM_JOB_ID:-local}  partition=${SLURM_JOB_PARTITION:-?}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python "$SCRIPT_DIR/infer_tma.py" "$@"
echo "[infer_tma.sbatch] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
