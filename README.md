# symplectic-RKF

This repo has the geodesic integrator from Bernard Ymeri's MQP, [*Developing a 
symplectic RKF integration scheme for calculating the geodesic equations on a 
manifold with applications to general 
relativity*](https://digital.wpi.edu/concern/student_works/p26770734?locale=en) 
(WPI, April 2026, advised by William Sanguinet), in two forms:

1. **Python**: the code transcribed from Appendix A
2. **C + GLSL**: the same algorithm as an OpenGL compute shader. It runs one
   geodesic per GPU thread, so it can do many initial conditions at once

The method itself: at each step, take an implicit symplectic RK4 step and an
implicit symplectic RK6 step (Gauss methods with 2 and 3 stages), each solved
by fixed-point iteration. If the two agree to within `tol`, keep the RK4
step. If they don't, shrink `h` once using the RKF formula and try again.

## What's here

| Path | What it is |
|---|---|
| `python/bernard_appendix_a.py` | Appendix A, copied exactly, comments included. Kerr black hole, `a = 0.5`, `r0 = 20`. |
| `python/bernard_schwarzschild.py` | The same file with `a = 0`, `r0 = 15` and the plot title changed, which is the setup in Figure 6. |
| `python/run_headless.py` | Runs either script unchanged on a node with no display: it saves the plot instead of showing it, and saves the trajectory as `.npy`. |
| `glsl/geodesic.comp` | The compute shader. Every function matches one in Appendix A and carries Bernard's comments. |
| `glsl/main.c` | Sets up OpenGL without a display, runs the shader, and writes the trajectories to `.npy`. |
| `environment.yaml` | The conda environment, with every package pinned to an exact version and build. |
| `plot_figures.py` | Bernard's plotting function, used to draw his figures from either version's output. |
| `paper_figures/` | Figures 6(a) and 7(a) as they appear in the MQP PDF, for comparison. |
| `turing/` | Slurm scripts for both versions. |

## Running on Turing

Everything runs in one conda environment: Python and the plotting packages,
plus the `libEGL.so.1` loader the GLSL version needs. Turing's GPU nodes have
NVIDIA's OpenGL driver but not that loader, so it comes from the environment
instead. Create it once, on the login node:

```bash
cd symplectic-RKF
module load miniconda3
conda init
conda env create -f environment.yaml
```

`environment.yaml` pins all 75 packages to exact versions and builds from
conda-forge (Python 3.11.17, numpy 2.4.6, matplotlib 3.10.9, numba 0.68.0,
libegl 1.7.0), so it builds the same environment every time. The batch
scripts activate it themselves.

### Python

```bash
sbatch turing/python.sbatch
```

This runs both scripts and puts the plots in `figures/python/`. Each run takes
a few minutes. To run one by hand:

```bash
conda activate symplectic-rkf
python python/run_headless.py python/bernard_appendix_a.py results/kerr_python
```

On a computer with a screen, `python python/bernard_appendix_a.py` opens the
plot window.

### C + GLSL

```bash
sbatch turing/glsl.sbatch
```

The job builds the program with the system C compiler, runs the Kerr and
Schwarzschild cases, and puts the plots in `figures/glsl/`. Near the top of
the log, `GL_RENDERER` should name the NVIDIA card.

To run it by hand on a GPU node (`srun --gres=gpu:1 --pty bash`):

```bash
conda activate symplectic-rkf
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
make -C glsl
./glsl/geodesic --out results/kerr_glsl.npy
python plot_figures.py --kerr results/kerr_glsl.npy --outdir figures/glsl
```

With the environment active, the program finds `libEGL.so.1` in it on its
own. `EGL_LIBRARY=/path/to/libEGL.so.1` points it at a different one.

`./glsl/geodesic --selftest` runs the shader's functions on the GPU (sin, cos,
division, the metric, its derivatives, the RK steps) and checks each against
the same code in plain C on the CPU. The batch job runs it first, so if a GPU
ever gets something wrong, the log says which function.

`./glsl/geodesic` with no options runs Appendix A's Kerr case. The options are
`--M`, `--a`, `--h`, `--steps`, `--ic "t r theta phi p_t p_r p_theta p_phi"`
(repeat it for more geodesics), `--ic-file` (one initial condition per line),
`--out`, `--chunk` and `--shader`. The output has shape
`(n_geodesics, steps + 1, 8)`. If a node has more than one GPU,
`EGL_DEVICE=n` picks which one to use.

## Reproducing the figures in the paper

`plot_figures.py` draws two versions of each plot: one from the code as
printed, and one that matches the figure in the PDF.

- **Kerr, Figure 7(a).** Appendix A runs 15,000 steps, but the published figure 
  doesn't show that many orbits. It matches the same run stopped after 7,823 steps. 
  That number comes from comparing against the published image; the paper doesn't 
  state it.
- **Schwarzschild, Figure 6(a).** This one is the first 5,220 steps, and
  Bernard rotated the 3D view by hand before saving it. The angle that best
  matches the published image is elevation 29.23°, azimuth 131.01°, roll 5.7°.

The axis labels look very slightly different because the paper's images came
from an older matplotlib. The trajectories themselves are the same.

Figures 1–5 (cylinder, sphere and torus) aren't included. Appendix A doesn't
have the code for them, and the paper doesn't give their step size or number
of steps.

## How the GLSL version compares to the Python

Over 15,000 steps the GLSL results stay within 1e-6 of the Python for Kerr and
1e-5 for Schwarzschild. Both keep the Hamiltonian constant to within a
few times 1e-10, and the plots from the two are identical.

They aren't bit-for-bit identical. The fixed-point iteration stops once
changes drop below 5e-11, so a difference in the last digit of a `sin` or a
sum can make it stop one iteration earlier or later. That shows up as
differences around 1e-11 per step, which add up to the 1e-6 above.

A few things in the shader work differently from the Python, but give the same
results:

- GLSL has no double precision `sin`, `cos` or `pow`, so the shader has its
  own. `sin` and `cos` follow FreeBSD's libm. The step-size update needs
  `(tol/error)**(1/5)`, which is computed with Newton's method.
- Division and square roots are done by hand too (`ddiv` and `dsqrt`). GLSL
  only promises they're as accurate as single precision, and NVIDIA's driver
  uses a fast approximate division. That wrecks the finite differences in
  `dgdx` (the metric at `r ± 1e-6` comes out the same), and on the L40S the
  orbits flew off in a straight line. The shader now starts from a single
  precision guess and refines it with Newton's method using `fma`, which gives
  exactly rounded double division on any GPU.
- Python calls `f(y)` two or three times for the starting guesses. The shader
  calls it once and copies the result, since the answer is always the same.
- The 15,000 steps are split across several GPU dispatches (`--chunk`, 100 by
  default), so no single dispatch runs long enough to hit a GPU timeout. This
  doesn't change the results.
