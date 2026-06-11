#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 benchmark grid: 1000 walk-100 instances, 14 heuristics x 6 algorithms.
# One array task per heuristic shard (see configs/cluster_walk1000.yaml for the
# shard map: 0=pdb, 1=regressor, 2-13=classifier threshold x temperature grid).
#
# CPU-only: at batch=1 the NN forward passes are <=4 states, where CPU beats
# GPU ~7x. partition=gpu-rtx3090 WITHOUT --gres schedules onto CPU nodes
# (proven by the phase-1 dedup job).
#
# Submit from phase3_experiments/:
#   sbatch jobs/run_grid_walk1000.sh
#
# Each shard writes results/cluster/walk1000_hNN.csv (resumable: resubmitting
# the array skips completed runs). Merge + summarize when all tasks finish:
#   python -m benchmark.summarize results/cluster/walk1000_h*.csv \
#       --csv results/cluster/walk1000_summary.csv
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=p3-grid
#SBATCH --array=0-13
#SBATCH --partition=gpu-rtx3090
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=jobs/logs/%A_%a.out
#SBATCH --error=jobs/logs/%A_%a.err

set -euo pipefail

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  task=$SLURM_ARRAY_TASK_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate search
echo "[job] python=$(which python)"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p jobs/logs results/cluster

SHARD=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
python run_experiments.py \
    --config configs/cluster_walk1000.yaml \
    --only-heuristic "$SLURM_ARRAY_TASK_ID" \
    --out "results/cluster/walk1000_h${SHARD}.csv"

echo "[job] done $(date)"
