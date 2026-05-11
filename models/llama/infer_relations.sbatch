#!/usr/bin/env bash
#SBATCH --job-name=llama-infer-relations
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# GPU-partition wrapper around models/llama/infer_relations.py.
# Partition is NOT set in the header (variable expansion does not work inside
# #SBATCH lines on most clusters). Submitter must pass --partition= on the
# sbatch command line:
#
#   sbatch --partition="$SLURM_PARTITION_GPU" models/llama/infer_relations.sbatch \
#       --input data/intermediate/tma_subset_events.jsonl \
#       --output data/intermediate/tma_subset_relations.jsonl \
#       --condition causal --sample-size 3
#
# All CLI args are forwarded to infer_relations.py. See UNI-66 spec §3.3, §6, §7.3.

set -euo pipefail

: "${SLURM_PARTITION_GPU:?set SLURM_PARTITION_GPU before sbatch (e.g. export SLURM_PARTITION_GPU=gpu_a100)}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$REPO_ROOT"   # so `python -m models.llama.infer_relations` resolves the package root

# SLURM does not source ~/.bashrc, so load the EESSI Python module the venv depends on.
module load 2025
module load Python/3.13.5-GCCcore-14.3.0

# shellcheck disable=SC1091
. "$REPO_ROOT/.venv/bin/activate"

echo "[infer_relations.sbatch] job=${SLURM_JOB_ID:-local}  partition=${SLURM_JOB_PARTITION:-?}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -m models.llama.infer_relations "$@"
echo "[infer_relations.sbatch] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
