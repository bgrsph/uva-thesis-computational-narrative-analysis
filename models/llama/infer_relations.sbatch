#!/usr/bin/env bash
#SBATCH --job-name=llama-infer-relations
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

: "${SLURM_PARTITION_GPU:?set SLURM_PARTITION_GPU before sbatch (e.g. export SLURM_PARTITION_GPU=gpu_a100)}"

# Under SLURM, $BASH_SOURCE points to /var/spool/slurm/slurmd/... (SLURM copies the
# script to a tmp dir before executing). Use $SLURM_SUBMIT_DIR (the directory sbatch
# was invoked from, which we require to be the repo root) and fall back to BASH_SOURCE
# for non-SLURM testing.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )}"

cd "$REPO_ROOT"   # so `python -m models.llama.infer_relations` resolves the package root

# SLURM does not source ~/.bashrc, so load the EESSI Python module the venv depends on.
module load 2025
module load Python/3.13.5-GCCcore-14.3.0

# shellcheck disable=SC1091
. "$REPO_ROOT/.venv/bin/activate"

echo "[infer_relations.sbatch] job=${SLURM_JOB_ID:-local}  partition=${SLURM_JOB_PARTITION:-?}  submit_dir=$REPO_ROOT  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -m models.llama.infer_relations "$@"
echo "[infer_relations.sbatch] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
