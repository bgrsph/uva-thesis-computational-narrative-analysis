#!/usr/bin/env bash
#SBATCH --job-name=embed-encode
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# GPU-partition wrapper around models/embed/encode_embeddings.py.
# Partition is NOT set in the header (variable expansion does not work inside
# #SBATCH lines on most clusters). Submitter must pass --partition= on the
# sbatch command line.
#
# Submit one job per encoder. The merger (src/build_embeddings.py) runs locally
# after all encoder jobs finish.
#
# Output path pattern: $EXPERIMENT_DIR/embeddings/<encoder>.jsonl
#   (filenames are explicit — see UNI-14 design).
#
# Example (e5_mistral):
#   sbatch --partition="$SLURM_PARTITION_GPU" models/embed/encode_embeddings.sbatch \
#       --encoder e5_mistral \
#       --input  $EXPERIMENT_DIR/experiment.jsonl \
#       --output $EXPERIMENT_DIR/embeddings/e5_mistral.jsonl

set -euo pipefail

: "${SLURM_PARTITION_GPU:?set SLURM_PARTITION_GPU before sbatch (e.g. export SLURM_PARTITION_GPU=gpu_a100)}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )}"

cd "$REPO_ROOT"

module load 2025
module load Python/3.13.5-GCCcore-14.3.0

# shellcheck disable=SC1091
. "$REPO_ROOT/.venv/bin/activate"

echo "[encode_embeddings.sbatch] job=${SLURM_JOB_ID:-local}  partition=${SLURM_JOB_PARTITION:-?}  submit_dir=$REPO_ROOT  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -m models.embed.encode_embeddings "$@"
echo "[encode_embeddings.sbatch] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
