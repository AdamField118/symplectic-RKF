#!/bin/bash
#SBATCH --job-name=srkf-python
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_outfiles/slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load miniconda3
conda activate symplectic-rkf
export MPLBACKEND=Agg
mkdir -p results

python python/run_headless.py python/bernard_appendix_a.py results/kerr_python
python python/run_headless.py python/bernard_schwarzschild.py results/schwarzschild_python

python plot_figures.py \
    --kerr results/kerr_python.npy --schwarzschild results/schwarzschild_python.npy --outdir figures/python
