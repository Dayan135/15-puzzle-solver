#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Train Phase 2 heuristic model on the cluster.
#
# Must be submitted from phase2_model_training/:
#   cd phase2_model_training
#   MODEL=classifier sbatch jobs/train_phase2.sh
#   MODEL=regressor  sbatch jobs/train_phase2.sh
#
# Optional epoch override:
#   MODEL=classifier EPOCHS=30 sbatch jobs/train_phase2.sh
#
# All other hyperparameters (lr, batch_size, tau, seed, …) are read from
# configs/{MODEL}.yaml — edit that file before submitting.
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=p2-train
#SBATCH --partition=rtx3090
#SBATCH --account=azencot
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=jobs/logs/%j.out
#SBATCH --error=jobs/logs/%j.err

set -euo pipefail

MODEL="${MODEL:-classifier}"
EPOCHS="${EPOCHS:-}"   # empty = use YAML default (classifier=14, regressor=10)

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME  model=$MODEL"

cd "$SLURM_SUBMIT_DIR"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate search
echo "[job] python=$(which python)"

# Abort immediately if no GPU allocated — catches silent CPU-only node assignment
if ! nvidia-smi --query-gpu=name --format=csv,noheader > /dev/null 2>&1; then
    echo "[error] No GPU detected on $SLURMD_NODENAME. Check --partition and --gres."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p jobs/logs

CONFIG="configs/${MODEL}.yaml"
echo "[job] config=$CONFIG  epochs=$EPOCHS"

python train.py \
  --config      "$CONFIG" \
  ${EPOCHS:+--epochs "$EPOCHS"} \
  --num-workers 8

echo "[job] done $(date)"
