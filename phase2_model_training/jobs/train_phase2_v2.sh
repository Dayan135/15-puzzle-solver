#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Train Phase 2 run-4 heuristic model (v2: 272-dim input + residual target).
#
# Must be submitted from phase2_model_training/:
#   cd phase2_model_training
#   MODEL=classifier sbatch jobs/train_phase2_v2.sh
#   MODEL=regressor  sbatch jobs/train_phase2_v2.sh
#
# Optional overrides:
#   MODEL=classifier EPOCHS=25 sbatch jobs/train_phase2_v2.sh
#
# Key differences from train_phase2.sh:
#   - Calls train_v2.py (not train.py)
#   - Uses configs/{MODEL}_v2.yaml
#   - Checkpoints → checkpoints_v2/  (overrides run-3 weights)
#   - Results     → results/run4/
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=p2-train-v2
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
EPOCHS="${EPOCHS:-}"

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME  model=$MODEL"

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

CONFIG="configs/${MODEL}_v2.yaml"
echo "[job] config=$CONFIG  epochs=${EPOCHS:-<from yaml>}"

python train_v2.py \
  --config      "$CONFIG" \
  ${EPOCHS:+--epochs "$EPOCHS"} \
  --num-workers 8

echo "[job] done $(date)"
