#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Evaluate a trained PuzzleClassifier across a CDF threshold × temperature grid.
#
# Must be submitted from phase2_model_training/:
#   cd phase2_model_training
#   sbatch jobs/evaluate_thresholds.sh
#
# Environment variable overrides (all optional):
#   CHECKPOINT  path to .pt file  (default: checkpoints/classifier/best_model.pt)
#   DATA        path to .bin file (default: cluster dataset path)
#   SPLIT       val | test         (default: test)
#   THRESHOLDS  space-separated   (default: 0.05 0.10 … 0.50)
#   TEMPERATURES space-separated  (default: 1.0 1.5 2.0 3.0)
#   OUT_DIR     results directory  (default: results/classifier)
#
# Examples:
#   # Default run (test split, full grid, best checkpoint)
#   sbatch jobs/evaluate_thresholds.sh
#
#   # Val split for threshold selection (before reporting test numbers)
#   SPLIT=val sbatch jobs/evaluate_thresholds.sh
#
#   # Evaluate a specific epoch snapshot
#   CHECKPOINT=checkpoints/classifier/epoch_010_mae1.013.pt \
#   OUT_DIR=results/classifier_epoch010 \
#   sbatch jobs/evaluate_thresholds.sh
#
#   # Fine-grained threshold sweep around 99% admissibility target
#   THRESHOLDS="0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10" \
#   sbatch jobs/evaluate_thresholds.sh
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=p2-eval-thresh
#SBATCH --partition=rtx3090
#SBATCH --account=azencot
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/%j.out
#SBATCH --error=jobs/logs/%j.err

set -euo pipefail

CHECKPOINT="${CHECKPOINT:-checkpoints/classifier/best_model.pt}"
DATA="${DATA:-/home/aviramom/projects/15-puzzle-solver/phase1_data_generation/data/full/dataset_000.bin}"
SPLIT="${SPLIT:-test}"
OUT_DIR="${OUT_DIR:-results/classifier}"

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"
echo "[job] checkpoint=$CHECKPOINT  split=$SPLIT  out=$OUT_DIR"

cd "$SLURM_SUBMIT_DIR"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate search
echo "[job] python=$(which python)"

if ! nvidia-smi --query-gpu=name --format=csv,noheader > /dev/null 2>&1; then
    echo "[error] No GPU detected on $SLURMD_NODENAME. Check --partition and --gres."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p jobs/logs

# Build optional threshold/temperature overrides
THRESH_ARGS=()
if [[ -n "${THRESHOLDS:-}" ]]; then
    THRESH_ARGS=(--thresholds $THRESHOLDS)
fi

TEMP_ARGS=()
if [[ -n "${TEMPERATURES:-}" ]]; then
    TEMP_ARGS=(--temperatures $TEMPERATURES)
fi

python evaluate_thresholds.py \
  --checkpoint  "$CHECKPOINT" \
  --data        "$DATA" \
  --split       "$SPLIT" \
  --out-dir     "$OUT_DIR" \
  --num-workers 8 \
  "${THRESH_ARGS[@]}" \
  "${TEMP_ARGS[@]}"

echo "[job] done $(date)"
