#!/bin/bash
#SBATCH --job-name=srkf-submit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=slurm_outfiles/slurm-%x-%j.out

# Submits the four versions as four separate jobs. Works as
# `sbatch turing/run_all.sbatch` or `bash turing/run_all.sbatch`, from the
# repository root. When they've all finished, `python compare_results.py
# results` shows how far each is from the Python.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

for v in python glsl fortran jax; do
    echo "$v: job $(sbatch --parsable turing/$v.sbatch)"
done
