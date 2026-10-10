# symplectic-RKF

This repo has the geodesic integrator from Bernard Ymeri's MQP, [*Developing a 
symplectic RKF integration scheme for calculating the geodesic equations on a 
manifold with applications to general 
relativity*](https://digital.wpi.edu/concern/student_works/p26770734?locale=en) 
(WPI, April 2026, advised by William Sanguinet), in four forms:

1. **Python**: the code transcribed from Appendix A
2. **C + GLSL**: the same algorithm as an OpenGL compute shader. It runs one
   geodesic per GPU thread, so it can do many initial conditions at once
3. **Fortran**: a function-by-function port, with the geodesics split across
   CPU cores by OpenMP
4. **JAX**: Appendix A with only the changes JAX needs, run on a GPU or split
   across CPU cores

The C + GLSL, Fortran and JAX programs take the same command-line options and
write the same `.npy` file.

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
| `fortran/geodesic.f90` | The Fortran version, with Bernard's comments. |
| `jax/geodesic_jax.py` | The JAX version. |
| `environment.yaml` | The conda environment, with every package pinned to an exact version and build. |
| `plot_figures.py` | Bernard's plotting function, used to draw his figures from any version's output. |
| `compare_results.py` | Prints how far each version's trajectories are from the Python's. |
| `paper_figures/` | Figures 6(a) and 7(a) as they appear in the MQP PDF, for comparison. |
| `turing/run_all.sbatch` | The Slurm job that runs all four. |

## Running on Turing

Everything runs in one conda environment: Python and the plotting packages,
the C and Fortran compilers, JAX with CUDA support, and the `libEGL.so.1`
loader the GLSL version needs (Turing's GPU nodes have NVIDIA's OpenGL driver
but not that loader). Create it once, on the login node:

```bash
cd symplectic-RKF
module load miniconda3
conda init
CONDA_OVERRIDE_CUDA=12.9 conda env create -f environment.yaml
```

`CONDA_OVERRIDE_CUDA` is needed because the login node has no GPU, and JAX's
CUDA build otherwise won't install. `environment.yaml` pins every package to an
exact version and build from conda-forge (Python 3.11.17, numpy 2.4.6,
matplotlib 3.10.9, numba 0.68.0, libegl 1.7.0, gcc and gfortran 14.4.0, JAX
0.10.2 for CUDA 12.9), so it builds the same environment every time.

Then submit the one job, from the repository root:

```bash
sbatch turing/run_all.sbatch
```

It runs all four versions on Appendix A's Kerr case and Figure 6's
Schwarzschild case, writes `results/<case>_<version>.npy`, puts the plots in
`figures/<version>/`, and ends by printing how far each version is from the
Python. It asks for one GPU and 8 CPU cores. `JAX_PLATFORM=cpu sbatch
turing/run_all.sbatch` runs the JAX version on the CPU cores instead of the
GPU.

### The programs

The C + GLSL, Fortran and JAX programs all work the same way. With no options
they run Appendix A's Kerr case. The options are `--M`, `--a`, `--h`,
`--steps`, `--ic "t r theta phi p_t p_r p_theta p_phi"` (repeat it for more
geodesics), `--ic-file` (one initial condition per line), `--out` and
`--chunk` (how many steps each pass takes; it doesn't change the results).
Each writes a `.npy` file of shape `(n_geodesics, steps + 1, 8)` and prints
its progress the same way. To run one by hand, `conda activate
symplectic-rkf` first.

**Python.** `python python/run_headless.py python/bernard_appendix_a.py
results/kerr_python` runs Bernard's script unchanged and saves the plot and
the trajectory. On a computer with a screen, `python
python/bernard_appendix_a.py` opens the plot window. It takes a few minutes.

**C + GLSL**, on a GPU node (`srun --gres=gpu:1 --pty bash`):

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
make -C glsl
./glsl/geodesic --out results/kerr_glsl.npy
```

With the environment active, the program finds `libEGL.so.1` in it on its own
(`EGL_LIBRARY=/path/to/libEGL.so.1` points it at a different one). Near the
top of its output, `GL_RENDERER` should name the NVIDIA card. It also takes
`--shader`, and `EGL_DEVICE=n` picks a GPU when there's more than one.

**Fortran:**

```bash
make -C fortran
OMP_NUM_THREADS=8 ./fortran/geodesic --out results/kerr_fortran.npy
```

The geodesics are shared out between `OMP_NUM_THREADS` threads (all cores if
it isn't set). A single geodesic is a chain of steps that each depend on the
one before, so one geodesic runs on one core; several run side by side. In a
test with 16 geodesics, 4 threads were 3.6 times faster than 1, with
bit-for-bit the same results. One geodesic of 15,000 steps takes about a
second.

**JAX:**

```bash
python jax/geodesic_jax.py --platform gpu --out results/kerr_jax.npy
python jax/geodesic_jax.py --platform cpu --out results/kerr_jax.npy
```

`--platform gpu` runs on the GPU. `--platform cpu` makes one JAX device per
CPU core (`SLURM_CPUS_PER_TASK`, or all of them) and splits the geodesics
between them. Either way the geodesics in a device's share run together, and
each `--chunk` of steps is one compiled program. Compiling takes about 30 s
the first time. On 4 CPU cores one geodesic then takes about 40 s; JAX pays
off with many geodesics, or on a GPU.

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

## How the versions compare to the Python

Over 15,000 steps, the largest difference from the Python in any of the eight
numbers was:

| | Kerr | Schwarzschild |
|---|---|---|
| C + GLSL | 1.6e-6 | 1.0e-5 |
| Fortran | 1.7e-6 | 2.9e-6 |
| JAX | 9.3e-7 | 9.8e-7 |

All of them keep the Hamiltonian constant to within a few times 1e-10, and
their plots are identical to the Python's (and match the figures in the paper
the same way). `compare_results.py` prints these numbers for a run.

They aren't bit-for-bit identical. The fixed-point iteration stops once
changes drop below 5e-11, so a difference in the last digit of a `sin` or a
sum can make it stop one iteration earlier or later. That shows up as
differences around 1e-11 per step, which add up to the 1e-6 above.

The Fortran version is the Python's code line for line, in order. The JAX
version changes only what JAX requires: `jnp` for `np`, `.at[].set()` instead
of writing into arrays, and `lax.while_loop` for the loops that `break` or
`return` early (with the same result on the iteration that stops). Each change
is marked `JAX:` in the code.

A few things in the shader work differently from the Python, but give the same
results:

- GLSL has no double precision `sin`, `cos` or `pow`, so the shader has its
  own. `sin` and `cos` follow FreeBSD's libm. The step-size update needs
  `(tol/error)**(1/5)`, which is computed with Newton's method.
- Division and square roots are done by hand too (`ddiv` and `dsqrt`). GLSL
  only promises they're as accurate as single precision, and NVIDIA's driver
  uses a fast approximate division. A division that rough wrecks the finite
  differences in `dgdx` (the metric at `r ± 1e-6` comes out the same) and the
  orbits fly off in a straight line. The shader starts from a single
  precision guess and refines it with Newton's method using `fma`, which gives
  exactly rounded double division on any GPU.
- The shader keeps positions and momenta as two `dvec4`s and the metric as a
  `dmat4` instead of arrays, and `dgdx(x, w)` returns the derivatives for one
  coordinate at a time instead of all four at once. On NVIDIA's driver the
  metric lost precision when it was kept in double arrays, and its compiler
  rejects writing to a vector component picked by a variable, so every index
  in the shader is a fixed number. The arithmetic is the same, term by term
  and in the same order.
- Python calls `f(y)` two or three times for the starting guesses. The shader
  calls it once and copies the result, since the answer is always the same.
- The 15,000 steps are split across several GPU dispatches (`--chunk`, 100 by
  default), so no single dispatch runs long enough to hit a GPU timeout. This
  doesn't change the results.
