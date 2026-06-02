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
#       --input      $EXPERIMENT_DIR/tma_subset.${SPLIT}.jsonl \
#       --output     $EXPERIMENT_DIR/tma_subset_events_full.${SPLIT}.jsonl \
#       --checkpoint data/intermediate/models/bert_crf \
#       --sample-size 10
#
# All CLI args are forwarded to infer_tma.py. See UNI-66 spec §3.1, §6, §7.1.

set -euo pipefail

: "${SLURM_PARTITION_CPU:?set SLURM_PARTITION_CPU before sbatch (e.g. export SLURM_PARTITION_CPU=rome)}"

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

echo "[infer_tma.sbatch] job=${SLURM_JOB_ID:-local}  partition=${SLURM_JOB_PARTITION:-?}  submit_dir=$REPO_ROOT  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python "$SCRIPT_DIR/infer_tma.py" "$@"
echo "[infer_tma.sbatch] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
