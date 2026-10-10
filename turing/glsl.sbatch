#!/bin/bash
#SBATCH --job-name=srkf-glsl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_outfiles/slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load miniconda3
conda activate symplectic-rkf
export MPLBACKEND=Agg
mkdir -p results

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# The GPU nodes have NVIDIA's EGL driver but no libEGL.so.1 loader. The binary
# uses the loader from the conda environment; this tells it where NVIDIA's
# driver is.
for f in /usr/share/glvnd/egl_vendor.d/10_nvidia.json /etc/glvnd/egl_vendor.d/10_nvidia.json; do
    if [[ -e "$f" ]]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES="$f"
        break
    fi
done

make -C glsl

# Appendix A: Kerr, M = 1, a = 0.5, q = (0, 20, pi/2, 0),
# p = (-0.9764550153430405, 0, 3.8, 3), h = 1, 15000 steps
./glsl/geodesic --out results/kerr_glsl.npy

# Figure 6: Schwarzschild (a = 0), r0 = 15, same momenta
./glsl/geodesic --a 0 \
    --ic "0 15 1.5707963267948966 0 -0.9764550153430405 0 3.8 3" \
    --out results/schwarzschild_glsl.npy

python plot_figures.py \
    --kerr results/kerr_glsl.npy --schwarzschild results/schwarzschild_glsl.npy --outdir figures/glsl
