#!/bin/bash
#SBATCH --job-name=srkf-video
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_outfiles/slurm-%x-%j.out

# Renders the black hole video to results/blackhole.mp4. Options after the
# script name go to the renderer, for example:
#     sbatch turing/visualization.sbatch --width 1920 --height 1080 --fps 60

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate symplectic-rkf
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

make -C visualization
./visualization/blackhole --out results/blackhole.mp4 "$@"
