#!/bin/bash
#SBATCH --job-name=puzzle-full
#SBATCH --partition=gpu-rtx3090
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
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

# ── step 1: additive 7-8 PDBs (one-shot; ~4 GB RAM for group B) ────────────────
mkdir -p ./data/pdbs
if [[ -f ./data/pdbs/pdb_a.bin && -f ./data/pdbs/pdb_b.bin ]]; then
    echo "[job] PDBs already present, skipping build_pdbs"
else
    ./build/build_pdbs --out ./data/pdbs
fi
echo "[job] pdbs ready"

# ── step 2: stratified dataset (uniform across cost buckets) ───────────────────
mkdir -p ./data/full
./build/generate_data \
    --threads  "$SLURM_CPUS_PER_TASK" \
    --target   100000000 \
    --pdb-dir  ./data/pdbs \
    --out-dir  ./data/full \
    --seed     "$SLURM_JOB_ID"

echo "[job] done $(date)"
echo "[job] output -> ./data/full/"
