#!/bin/bash
#SBATCH --job-name=puzzle-pdbs
#SBATCH --partition=gpu-rtx3090
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/%j.out
#SBATCH --error=jobs/logs/%j.err

set -euo pipefail

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

# ── build ─────────────────────────────────────────────────────────────────────
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$SLURM_CPUS_PER_TASK" --target build_pdbs
echo "[job] build done"

# ── run: one-shot additive 7-8 PDB build (CPU-only; ~4 GB RAM for group B) ──────
mkdir -p ./data/pdbs
if [[ -f ./data/pdbs/pdb_a.bin && -f ./data/pdbs/pdb_b.bin ]]; then
    echo "[job] PDBs already present, nothing to do"
else
    ./build/build_pdbs --out ./data/pdbs
fi

echo "[job] pdbs -> ./data/pdbs/"
ls -l ./data/pdbs/
echo "[job] done $(date)"
