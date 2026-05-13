#!/usr/bin/env bash
#SBATCH --job-name=bertcrf-infer-tma-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# GPU-partition wrapper around models/bert_crf/infer_tma.py.
# Mirrors infer_tma.sbatch but targets a CUDA GPU; the Python script's
# _pick_device() picks "cuda" automatically when available.
#
# Partition is NOT set in the header (variable expansion does not work inside
# #SBATCH lines on most clusters). Submitter must pass --partition= on the
# sbatch command line:
#
#   sbatch --partition="$SLURM_PARTITION_GPU" models/bert_crf/infer_tma.gpu.sbatch \
#       --input      data/intermediate/tma_subset.jsonl \
#       --output     data/intermediate/tma_subset_events_full.jsonl \
#       --checkpoint data/intermediate/models/bert_crf \
#       --batch-size 128

set -euo pipefail

: "${SLURM_PARTITION_GPU:?set SLURM_PARTITION_GPU before sbatch (e.g. export SLURM_PARTITION_GPU=gpu_a100)}"

# Under SLURM, $BASH_SOURCE points to /var/spool/slurm/slurmd/... (SLURM copies the
# script to a tmp dir before executing). Use $SLURM_SUBMIT_DIR (the directory sbatch
# was invoked from, which we require to be the repo root) and fall back to BASH_SOURCE
# for non-SLURM testing.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )}"
SCRIPT_DIR="$REPO_ROOT/models/bert_crf"

# SLURM does not source ~/.bashrc, so load the EESSI Python module the venv depends on.
module load 2025
module load Python/3.13.5-GCCcore-14.3.0

# shellcheck disable=SC1091
. "$REPO_ROOT/.venv-maven-train/bin/activate"

# Fail fast if torch can't see a CUDA device — otherwise _pick_device() falls back
# to CPU silently and we'd burn a GPU allocation running on the host CPU.
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible — check torch build in .venv-maven-train'"

echo "[infer_tma.gpu.sbatch] job=${SLURM_JOB_ID:-local}  partition=${SLURM_JOB_PARTITION:-?}  submit_dir=$REPO_ROOT  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python "$SCRIPT_DIR/infer_tma.py" "$@"
echo "[infer_tma.gpu.sbatch] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
