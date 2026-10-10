#!/bin/bash
#SBATCH --job-name=srkf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_outfiles/slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate symplectic-rkf
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p results

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# With no options the programs run Appendix A's Kerr case; this is Figure 6's
# Schwarzschild case.
SCHW=(--a 0 --ic "0 15 1.5707963267948966 0 -0.9764550153430405 0 3.8 3")

# Python: Bernard's code exactly as printed
python python/run_headless.py python/bernard_appendix_a.py results/kerr_python
python python/run_headless.py python/bernard_schwarzschild.py results/schwarzschild_python

# C + GLSL. The GPU nodes have NVIDIA's EGL driver but no libEGL.so.1 loader;
# the binary uses the loader from the conda environment, and this tells it
# where NVIDIA's driver is.
for f in /usr/share/glvnd/egl_vendor.d/10_nvidia.json /etc/glvnd/egl_vendor.d/10_nvidia.json; do
    if [[ -e "$f" ]]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES="$f"
        break
    fi
done
make -C glsl
./glsl/geodesic --out results/kerr_glsl.npy
./glsl/geodesic "${SCHW[@]}" --out results/schwarzschild_glsl.npy

# Fortran, on OMP_NUM_THREADS cores
make -C fortran
./fortran/geodesic --out results/kerr_fortran.npy
./fortran/geodesic "${SCHW[@]}" --out results/schwarzschild_fortran.npy

# JAX, on the GPU (or the CPU cores with JAX_PLATFORM=cpu)
python jax/geodesic_jax.py --platform "${JAX_PLATFORM:-gpu}" --out results/kerr_jax.npy
python jax/geodesic_jax.py --platform "${JAX_PLATFORM:-gpu}" "${SCHW[@]}" --out results/schwarzschild_jax.npy

for v in python glsl fortran jax; do
    python plot_figures.py \
        --kerr results/kerr_$v.npy --schwarzschild results/schwarzschild_$v.npy --outdir figures/$v
done
python compare_results.py results
