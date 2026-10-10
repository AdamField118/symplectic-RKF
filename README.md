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

`visualization/` puts the integrator to work on something else: a video of a
spinning black hole, with every pixel's light ray traced on the GPU.

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
| `visualization/` | The black hole video: `blackhole.comp` traces the light rays, `render.c` runs it and hands the frames to ffmpeg. |
| `turing/` | A Slurm job for each version, `run_all.sbatch`, which submits all four, and `visualization.sbatch` for the video. |

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
0.10.2 for CUDA 12.9, ffmpeg 5.1), so it builds the same environment every
time. ffmpeg is there for the video; newer versions of it need libraries that
clash with the ones JAX uses.

Each version has its own Slurm job. Submit from the repository root:

```bash
sbatch turing/python.sbatch    # 1 CPU core, a few minutes
sbatch turing/glsl.sbatch      # 1 GPU
sbatch turing/fortran.sbatch   # 8 CPU cores
sbatch turing/jax.sbatch       # 1 GPU (8 CPU cores with JAX_PLATFORM=cpu)
```

or all four at once, as four separate jobs:

```bash
sbatch turing/run_all.sbatch   # or: bash turing/run_all.sbatch
```

Each job runs its version on Appendix A's Kerr case and Figure 6's
Schwarzschild case, writes `results/<case>_<version>.npy`, and puts the plots
in `figures/<version>/`. `JAX_PLATFORM=cpu sbatch turing/jax.sbatch` runs the
JAX version on the CPU cores instead of the GPU. Once they've finished,
`python compare_results.py results` prints how far each version is from the
Python.

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
each `--chunk` of steps is one compiled program, compiled once and reused for
every chunk (the log prints how long compiling took). On a CPU, one geodesic
takes about 4 s in all: about 3 s to compile, then about 1.3 s for the 15,000
steps.

Python loops inside JAX code are unrolled when it's compiled, so Bernard's
loops over the metric's entries, the four coordinates in `dgdx`, and the
64-term momentum sum are written as whole-array operations instead (same
arithmetic, same order). Written as loops, they made the compiled program 15
times bigger, took 14 s to compile, and launched about 1,500 small operations
per step, which is what made the GPU version slow.

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
| JAX | 8.2e-7 | 9.8e-7 |

All of them keep the Hamiltonian constant to within a few times 1e-10, and
their plots are identical to the Python's (and match the figures in the paper
the same way). `compare_results.py` prints these numbers for a run.

They aren't bit-for-bit identical. The fixed-point iteration stops once
changes drop below 5e-11, so a difference in the last digit of a `sin` or a
sum can make it stop one iteration earlier or later. That shows up as
differences around 1e-11 per step, which add up to the 1e-6 above.

The Fortran version is the Python's code line for line, in order. The JAX
version changes only what JAX requires: `jnp` for `np`, `lax.while_loop` for
the loops that `break` or `return` early (with the same result on the
iteration that stops), and whole-array operations instead of loops that write
one entry at a time (above). Each change is marked `JAX:` in the code. XLA,
the compiler under JAX, also fuses `a*b + c` into a single fused multiply-add,
rounding once where NumPy rounds twice, and where it does so depends on how
the program is laid out. That's why JAX's numbers differ slightly between
runs on different numbers of devices.

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

## Black hole video

`visualization/` renders a video of a Kerr black hole and its accretion disk.
Each pixel is a light ray, traced backwards from the camera with Bernard's
RK4/RK6 integrator in a compute shader, one ray per GPU thread. The look and
the camera's path come from
[blackhole-bg](https://codeberg.org/AdamField118/blackhole-bg): over 8
seconds the camera sets off up and to the left out of the disk, circles the
hole once and comes back to where it started, so the video loops.
blackhole-bg's hole doesn't spin and uses its own integrator; this one is
Appendix A's Kerr black hole (`a = 0.5`), so the shadow is lopsided, and the
side of the disk coming towards the camera is brighter and whiter than the side
going away.

```bash
sbatch turing/visualization.sbatch                # results/blackhole.mp4, 1280x720 at 30 fps
sbatch turing/visualization.sbatch --width 1920 --height 1080 --fps 60
```

or by hand on a GPU node, with the environment active:

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
make -C visualization
./visualization/blackhole --out results/blackhole.mp4
./visualization/blackhole --frame 60 --out frame.png   # just one frame
```

The options are listed at the top of `visualization/render.c`. `--a 0` gives
blackhole-bg's non-spinning hole, `--no-doppler` its colours, `--no-disk`
just the hole and the stars.

The program prints how long each frame took, and at the end how many frames
per second it rendered and how that compares to real time. ffmpeg encodes on
the CPU cores while the GPU draws, so the job asks for 8 of them. For more
speed, lower the resolution or raise `--step`: `--step 0.15` was about 1.6
times faster than the default 0.08 in tests, and changed 0.5% of pixels
visibly.

To keep it fast, the shader uses the integrator a little differently from
Appendix A:

- It works in single precision, which GPUs run 32 to 64 times faster than
  double.
- The derivatives of the inverse metric are worked out exactly instead of by
  finite differences. In single precision `r + 1e-6` rounds back to `r`, and
  the exact derivatives take one evaluation of the metric instead of eight.
  They agree with Appendix A's `dgdx` to within its finite-difference error.
- The tolerances are larger (`--tol 1e-3`, `--rk-tol 1e-5`), since single
  precision can't get to 5e-11.
- Each step starts with `h` proportional to the distance from the hole, so
  rays take small steps close to it and big ones far away. The RK4/RK6
  comparison then shrinks the step once if it needs to, as in Appendix A.
- A ray stops when it heads inwards inside the innermost photon orbit, since
  it can't come back out from there. Going on to the horizon itself made a
  few rays come out wrong, because the equations get very stiff there.

Two checks that the rays go where they should. With `a = 0`, the shadow seen
from 20M has an angular radius of 14.270°; the exact value is 14.269°. With
`a = 0.5` and `a = 0.9`, which rays fall in matches, pixel for pixel, a run
with 8 times smaller steps and a 100 times smaller `--tol`.

With no `--sky` the stars are generated. To use the Hubble image blackhole-bg
uses, bake it the same way (or use the `starfield.rgb` from a blackhole-bg
build):

```bash
curl -LO https://esahubble.org/media/archives/images/publicationtiff/heic1112d.tif
ffmpeg -i heic1112d.tif -frames:v 1 -pix_fmt rgb24 -f rawvideo \
    -vf "scale=2048:2048:force_original_aspect_ratio=increase,crop=2048:2048" starfield.rgb
./visualization/blackhole --sky starfield.rgb --out results/blackhole.mp4
```
