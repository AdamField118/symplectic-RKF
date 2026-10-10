#!/bin/bash
#SBATCH --job-name=srkf-jax
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_outfiles/slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate symplectic-rkf
export MPLBACKEND=Agg
mkdir -p results

export XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# With no options the programs run Appendix A's Kerr case; this is Figure 6's
# Schwarzschild case.
SCHW=(--a 0 --ic "0 15 1.5707963267948966 0 -0.9764550153430405 0 3.8 3")

# On the GPU, or on the CPU cores with JAX_PLATFORM=cpu sbatch turing/jax.sbatch
python jax/geodesic_jax.py --platform "${JAX_PLATFORM:-gpu}" --out results/kerr_jax.npy
python jax/geodesic_jax.py --platform "${JAX_PLATFORM:-gpu}" "${SCHW[@]}" --out results/schwarzschild_jax.npy

python plot_figures.py \
    --kerr results/kerr_jax.npy --schwarzschild results/schwarzschild_jax.npy --outdir figures/jax
