#!/bin/bash
#SBATCH --job-name=srkf-fortran
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_outfiles/slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate symplectic-rkf
export MPLBACKEND=Agg
mkdir -p results

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# With no options the programs run Appendix A's Kerr case; this is Figure 6's
# Schwarzschild case.
SCHW=(--a 0 --ic "0 15 1.5707963267948966 0 -0.9764550153430405 0 3.8 3")

make -C fortran
./fortran/geodesic --out results/kerr_fortran.npy
./fortran/geodesic "${SCHW[@]}" --out results/schwarzschild_fortran.npy

python plot_figures.py \
    --kerr results/kerr_fortran.npy --schwarzschild results/schwarzschild_fortran.npy --outdir figures/fortran
