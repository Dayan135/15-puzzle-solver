#!/bin/bash
#SBATCH --job-name=puzzle-full
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
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
mkdir -p ./data/full

./build/generate_data \
    --threads  "$SLURM_CPUS_PER_TASK" \
    --target   10000000 \
    --scramble 30 \
    --out-dir  ./data/full \
    --seed     "$SLURM_JOB_ID"

echo "[job] done $(date)"
echo "[job] output -> ./data/full/"
