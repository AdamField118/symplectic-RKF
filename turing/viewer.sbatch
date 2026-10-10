#!/bin/bash
#SBATCH --job-name=srkf-viewer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_outfiles/slurm-%x-%j.out

# Runs the live viewer until the time limit (or scancel). The address to open
# is near the top of the log:
#     grep -m1 ondemand slurm_outfiles/slurm-srkf-viewer-<job>.out
# Options after the script name go to viewer.py: a port, then renderer
# options, for example
#     sbatch turing/viewer.sbatch 4631 --sky starfield.rgb

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate symplectic-rkf

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
python -u visualization/viewer.py "$@"
