"""Run one of Bernard's scripts without changing it, on a machine with no display.

    python python/run_headless.py python/bernard_appendix_a.py results/kerr_python

The script runs exactly as written. The only difference is that plt.show()
saves the figure to OUT.png instead of opening a window, and afterwards the
script's `geodesic` array is saved to OUT.npy so plot_figures.py can use it.
"""
import runpy
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

script, out = sys.argv[1], sys.argv[2]
plt.show = lambda *args, **kwargs: plt.savefig(out + ".png")

start = time.perf_counter()
namespace = runpy.run_path(script, run_name="__main__")
print(f"{script} took {time.perf_counter() - start:.0f} s")

np.save(out + ".npy", np.asarray(namespace["geodesic"]))
print(f"wrote {out}.png and {out}.npy")
