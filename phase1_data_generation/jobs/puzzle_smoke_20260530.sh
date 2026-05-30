#!/bin/bash
#SBATCH --job-name=puzzle-smoke
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=jobs/logs/%j.out
#SBATCH --error=jobs/logs/%j.err

set -euo pipefail

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

# ── build ─────────────────────────────────────────────────────────────────────
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$SLURM_CPUS_PER_TASK"
echo "[job] build done"

# ── run ───────────────────────────────────────────────────────────────────────
mkdir -p ./data/smoke

./build/generate_data \
    --threads  "$SLURM_CPUS_PER_TASK" \
    --target   200 \
    --scramble 30 \
    --out-dir  ./data/smoke \
    --seed     "$SLURM_JOB_ID"

echo "[job] done $(date)"
echo "[job] output -> ./data/smoke/"
