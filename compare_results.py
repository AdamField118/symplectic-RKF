"""How far each version's trajectories are from Bernard's Python.

    python compare_results.py results

Reads results/<case>_<version>.npy for the Kerr and Schwarzschild cases and
prints the largest difference from results/<case>_python.npy.
"""
import os
import sys

import numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else "results"
for case in ["kerr", "schwarzschild"]:
    reference = np.load(os.path.join(folder, f"{case}_python.npy"))
    for version in ["glsl", "fortran", "jax"]:
        path = os.path.join(folder, f"{case}_{version}.npy")
        if not os.path.exists(path):
            print(f"{case:14s} {version:8s} (no results)")
            continue
        g = np.load(path)
        g = g[0] if g.ndim == 3 else g
        print(f"{case:14s} {version:8s} largest difference from the Python: {np.abs(g - reference).max():.1e}")
