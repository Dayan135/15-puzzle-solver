#!/bin/bash
#SBATCH --job-name=p2-test-data
#SBATCH --partition=gpu-rtx3090
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=jobs/logs/%j.out
#SBATCH --error=jobs/logs/%j.err

set -euo pipefail

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate search
echo "[job] python=$(which python)"

# Real Phase 1 dataset lives under phase1; reference it relative to the repo root.
DATA=../phase1_data_generation/data/full

python tests/test_dataset.py --data "$DATA"

echo "[job] done $(date)"
