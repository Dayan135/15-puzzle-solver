#!/bin/bash
#SBATCH --job-name=puzzle-dedup
#SBATCH --partition=gpu-rtx3090
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --output=jobs/logs/%j.out
#SBATCH --error=jobs/logs/%j.err

set -euo pipefail

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

# numpy lives in the Phase 2 conda env; no GPU needed for this job.
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate search
echo "[job] python=$(which python)"

python scripts/dedup_dataset.py ./data/full --out-dir ./data/dedup

echo "[job] done $(date)"
echo "[job] output -> ./data/dedup/"
ls -l ./data/dedup/
