"""Bernard Ymeri's symplectic RK4/RK6 geodesic integrator (WPI MQP 2026,
Appendix A) in JAX.

The functions from inverse_metric() to symplectic_rk46_comparison() are
Bernard's, with only the changes JAX needs, each marked "JAX:" (jnp instead of
np, .at[].set() instead of in-place writes, lax.while_loop instead of loops
that break or return early). Everything is float64.

The command line and the output are the same as the GLSL and Fortran
versions': the result is a NumPy .npy file of shape (n_geodesics, steps + 1, 8).
Geodesics run side by side (jax.vmap), and the 15000 steps are compiled into
one lax.scan per --chunk.

    python jax/geodesic_jax.py [--platform gpu|cpu] [--M 1.0] [--a 0.5] [--h 1]
        [--steps 15000] [--chunk 100] [--ic "t r th ph pt pr pth pph"]...
        [--ic-file FILE] [--out geodesic.npy]

--platform gpu runs on the GPU(s). --platform cpu makes one JAX device per CPU
core (SLURM_CPUS_PER_TASK, or all of them) and splits the geodesics between
them. Without --platform, JAX picks the GPU if there is one. A single geodesic
is a chain of steps that each depend on the last, so it runs on one device.
"""
import argparse
import os
import sys
import time

import numpy as np

parser = argparse.ArgumentParser(description="Symplectic RK4/RK6 geodesic integrator (JAX)")
parser.add_argument("--platform", choices=["gpu", "cpu"])
parser.add_argument("--M", type=float, default=1.0)
parser.add_argument("--a", type=float, default=0.5)
parser.add_argument("--h", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=15000)
parser.add_argument("--chunk", type=int, default=100)
parser.add_argument("--ic", action="append", default=[])
parser.add_argument("--ic-file")
parser.add_argument("--out", default="geodesic.npy")
args = parser.parse_args()

# Devices have to be chosen before JAX is imported
if args.platform == "cpu":
    n_cpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + f" --xla_force_host_platform_device_count={n_cpu}")
elif args.platform == "gpu":
    os.environ["JAX_PLATFORMS"] = "cuda"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

# JAX: Bernard's code runs in float64 (NumPy default); JAX defaults to float32.
jax.config.update("jax_enable_x64", True)



# Kerr metric set up
# (set from --M and --a below, before anything is compiled)
M = 1.0
a = 0.5



# Inverse Kerr metric (uses the inverse metric like the Fantasty code). Takes in
#a vector of length 4 representing the 4 Boyder-Lindquist coordinates: t, r,
#theta, and phi
def inverse_metric(x):

    t = x[0]
    r = x[1]
    theta = x[2]
    phi = x[3]

    #Making it easier and more compact to write out the full Kerr metric
    Sigma = r**2 + a**2 * jnp.cos(theta)**2
    Delta = r**2 - 2*M*r + a**2

    #Set up a 4x4 matrix of zeroes
    g = jnp.zeros((4, 4))

    #Input all the spacetime metric components (Kerr for this example)
    g = g.at[0, 0].set(-((r**2 + a**2)**2 - a**2 * Delta * jnp.sin(theta)**2) / (Delta * Sigma))
    g = g.at[0, 1].set(0.0)
    g = g.at[0, 2].set(0.0)
    g = g.at[0, 3].set(-2 * M * a * r / (Delta * Sigma))

    g = g.at[1, 0].set(g[0, 1])
    g = g.at[1, 1].set(Delta / Sigma)
    g = g.at[1, 2].set(0.0)
    g = g.at[1, 3].set(0.0)

    g = g.at[2, 0].set(g[0, 2])
    g = g.at[2, 1].set(g[1, 2])
    g = g.at[2, 2].set(1.0 / Sigma)
    g = g.at[2, 3].set(0.0)

    g = g.at[3, 0].set(g[0, 3])
    g = g.at[3, 1].set(g[1, 3])
    g = g.at[3, 2].set(g[2, 3])
    g = g.at[3, 3].set((Delta - a**2 * jnp.sin(theta)**2) / (Delta * Sigma * jnp.sin(theta)**2))

    return g



# Numerical derivatives of inverse metric
def dgdx(x, delta=1e-6):
    # Find the number of variables we're working with and create an n by n by n array
    #that will hold all the partial derivvatives of every entry of the inverse metric
    #matrix
    n = len(x)
    dg = jnp.zeros((n, n, n))

    # Uses central finite differences to approximate each partial derivative.
    #The n-th loop computes all 4 partial derivatives for the n-th spacetime
    #coordinate.
    for w in range(n):
        dx = jnp.zeros(n)
        dx = dx.at[w].set(delta)
        g_plus = inverse_metric(x + dx)
        g_minus = inverse_metric(x - dx)
        dg = dg.at[w].set((g_plus - g_minus) / (2 * delta))

    return dg



# Hamiltonian
def hamiltonian(y):
    # Takes the first n elements of the initial conditions vector y (these
    #correspond to the position coordinates) and the last n elements of y (these
    #correspond to the momentum coordinates), and makes them their own vectors.
    n = len(y) // 2
    x = y[:n]
    p = y[n:]

    g = inverse_metric(x)
    dg = dgdx(x)

    # Following the Hamiltonian equations for geodesics, we calculate the
    #derivatives of the position and momentum components
    dx = jnp.dot(g,p)

    dp = jnp.zeros(n)

    # Uses a triple for-loop to sum over the 3 indices in the equation for the
    #momentum derivative
    for w in range(n):
        for u in range(n):
            for v in range(n):
                dp = dp.at[w].set(dp[w] - 0.5 * dg[w, u, v] * p[u] * p[v])

    return jnp.concatenate([dx, dp])



# Symplectic RK4 coefficients
r4a11 = 1/4
r4a12 = 1/4 - np.sqrt(3)/6
r4a21 = 1/4 + np.sqrt(3)/6
r4a22 = 1/4

r4b1 = 1/2
r4b2 = 1/2



# Symplectic RK4 step, uses fixed point iteration to solve the implicit RK method
def symplectic_rk4(f, y, h, tol=5e-11, max_iter=20):
    # Gives the initial guess for k1 and k2 based on the input
    k1 = f(y)
    k2 = f(y)

    # Updates k1 and k2 based on the symplectic rk4 coefficients and the previous
    #guess for k1 and k2.
    # JAX: `for _ in range(max_iter): ... break` as a lax.while_loop. On the
    # iteration that breaks, k1/k2 keep their previous values, exactly as in the
    # Python loop (the break comes before the replacement).
    def cond(state):
        i, k1, k2, done = state
        return (i < max_iter) & ~done

    def body(state):
        i, k1, k2, done = state
        k1_new = f(y + h * (r4a11 * k1 + r4a12 * k2))
        k2_new = f(y + h * (r4a21 * k1 + r4a22 * k2))

        # Checks if the difference (in the regular 2-norm) of k1 and the updated
        #k1 is less than the given tolerance, and if the same is true for k2 and
        #the updated k2.
        done = ((jnp.linalg.norm(k1_new - k1) < tol) & (jnp.linalg.norm(k2_new - k2) < tol))

        # Replaces the previous guess with the new one
        k1 = jnp.where(done, k1, k1_new)
        k2 = jnp.where(done, k2, k2_new)
        return i + 1, k1, k2, done

    _, k1, k2, _ = jax.lax.while_loop(cond, body, (0, k1, k2, False))

    # Uses the final guesses and the symplectic rk4 coefficients to update y.
    return y + h * (r4b1 * k1 + r4b2 * k2)



# Symplectic RK6 coefficients
r6a11 = 5/36
r6a12 = 2/9 - np.sqrt(15)/15
r6a13 = 5/36 - np.sqrt(15)/30

r6a21 = 5/36 + np.sqrt(15)/24
r6a22 = 2/9
r6a23 = 5/36 - np.sqrt(15)/24

r6a31 = 5/36 + np.sqrt(15)/30
r6a32 = 2/9 + np.sqrt(15)/15
r6a33 = 5/36

r6b1 = 5/18
r6b2 = 4/9
r6b3 = 5/18



# Symplectic RK6 step, uses fixed point iteration to solve the implicit RK method
def symplectic_rk6(f, y, h, tol=5e-11, max_iter=20):
    # Gives the initial guess for k1, k2, and k3 based on the input
    k1 = f(y)
    k2 = f(y)
    k3 = f(y)

    # Checks if the difference (in the regular 2-norm) of k1 and the updated
    #k1 is less than the given tolerance, and if the same is true for k2 and
    #the updated k2, and k3 and the updated k3.
    # JAX: same for/break -> lax.while_loop translation as symplectic_rk4.
    def cond(state):
        i, k1, k2, k3, done = state
        return (i < max_iter) & ~done

    def body(state):
        i, k1, k2, k3, done = state
        k1_new = f(y + h * (r6a11*k1 + r6a12*k2 + r6a13*k3))
        k2_new = f(y + h * (r6a21*k1 + r6a22*k2 + r6a23*k3))
        k3_new = f(y + h * (r6a31*k1 + r6a32*k2 + r6a33*k3))

        # Convergence check (2-norm)
        done = (jnp.linalg.norm(k1_new - k1) < tol) & \
               (jnp.linalg.norm(k2_new - k2) < tol) & \
               (jnp.linalg.norm(k3_new - k3) < tol)

        # Replaces the previous guess with the new one
        k1 = jnp.where(done, k1, k1_new)
        k2 = jnp.where(done, k2, k2_new)
        k3 = jnp.where(done, k3, k3_new)
        return i + 1, k1, k2, k3, done

    _, k1, k2, k3, _ = jax.lax.while_loop(cond, body, (0, k1, k2, k3, False))

    # Uses the final guesses and the symplectic rk6 coefficients to update y.
    return y + h * (r6b1*k1 + r6b2*k2 + r6b3*k3)



# We'll compute an rk4 and an rk6 step, find the error between them, and decide
#whether to accept the rk4 step, or shrink the step size and try again until the
#error is less than a given tolerance.
def symplectic_rk46_comparison(f, y, h, tol=0.1, max_shrinks=1, rk_tol=5e-11, rk_max_iter=20):

    # Define the current default step size
    # JAX: as a float64 so the loop carry has one type (h is passed as the int 1).
    h_current = jnp.asarray(h, dtype=jnp.float64)

    # A for loop = computes an rk4 and rk6 step until the error between them is
    #below a given tolerance, then computes them again with a smaller step size
    #if it's not
    # JAX: the early `return y_rk4` becomes the `accepted` flag that ends the
    # lax.while_loop; falling out of the loop returns the last y_rk4, as before.
    def cond(state):
        i, h_current, y_rk4, accepted = state
        return (i < max_shrinks + 1) & ~accepted

    def body(state):
        i, h_current, y_rk4, accepted = state

        # Compute a symplectic rk4 step
        y_rk4 = symplectic_rk4(f, y, h_current, tol=rk_tol, max_iter=rk_max_iter)

        # Compute a symplectic rk6 step
        y_rk6 = symplectic_rk6(f, y, h_current, tol=rk_tol, max_iter=rk_max_iter)

        # Find the 2 norm of their difference
        error = jnp.linalg.norm(y_rk6 - y_rk4)

        # Accept symplectic rk4 step if error is below the tolerance
        accepted = error < tol

        # Update step size using formula that's used in the rkf method
        h_current = jnp.where(accepted, h_current, 0.9 * h_current * (tol / error)**(1/5))
        return i + 1, h_current, y_rk4, accepted

    _, _, y_rk4, _ = jax.lax.while_loop(cond, body, (0, h_current, jnp.zeros_like(y), False))

    # If the error still isn't below the tolerance after max_shrinks, accept the rk4 step
    return y_rk4



# Integrate Geodesic
# JAX: the steps of one geodesic are a lax.scan, `n` steps per call; the
# trajectory list is the scan's output.
def integrate_geodesic(y, h, n):
    # Compute the position/momentum vector using the symplectic rk4, adds it to the
    #trajectory list, and then computes the next position/momentum in the geodesic
    def step(y, _):
        y = symplectic_rk46_comparison(hamiltonian, y, h)
        return y, y

    return jax.lax.scan(step, y, None, length=n)


def main():
    global M, a
    M, a = args.M, args.a

    # Initial conditions, the same way the GLSL and Fortran versions read them
    ics = [[float(v) for v in s.split()] for s in args.ic]
    if args.ic_file:
        with open(args.ic_file) as f:
            ics += [[float(v) for v in line.split()] for line in f if len(line.split()) == 8]
    if not ics:
        # Appendix A's initial conditions
        ics = [[0.0, 20.0, np.pi / 2, 0.0, -0.9764550153430405, 0.0, 3.8, 3.0]]
    if any(len(ic) != 8 for ic in ics):
        sys.exit("error: --ic needs 8 numbers")
    ics = np.array(ics, dtype=np.float64)
    n_ic = len(ics)
    steps = args.steps
    chunk = max(1, min(args.chunk, steps)) if steps > 0 else 1

    # One JAX device per CPU core (or per GPU); the geodesics are split evenly
    # between them, padded with copies of the last one to fill every device.
    devices = jax.devices()
    print(f"JAX {jax.__version__} on {len(devices)} {devices[0].platform} device(s)", file=sys.stderr)
    n_dev = len(devices)
    per_dev = -(-n_ic // n_dev)
    y = np.concatenate([ics, np.repeat(ics[-1:], per_dev * n_dev - n_ic, axis=0)])
    mesh = Mesh(np.array(devices), ("g",))

    def advance(y, n):
        run = jax.vmap(lambda y0: integrate_geodesic(y0, args.h, n))
        # check_vma=False: each device works on its own geodesics and never
        # talks to the others, so JAX doesn't need to track which values differ
        # between devices (that tracking rejects Bernard's while_loops as written).
        return jax.jit(jax.shard_map(run, mesh=mesh, in_specs=P("g"), out_specs=(P("g"), P("g")),
                                     check_vma=False))

    traj = np.empty((n_ic, steps + 1, 8))
    traj[:, 0] = ics  # geodesic.append(y.copy()) for the starting point
    compiled = {}
    t0 = time.perf_counter()
    done = 0
    # Lay the geodesics out across the devices up front, the same way each
    # chunk hands them back, so every chunk reuses one compiled program.
    y = jax.device_put(jnp.asarray(y), NamedSharding(mesh, P("g")))
    while done < steps:
        c = min(chunk, steps - done)
        if c not in compiled:
            compiled[c] = advance(y, c)
        y, rows = compiled[c](y)
        traj[:, done + 1:done + 1 + c] = np.asarray(rows)[:n_ic]
        done += c
        print(f"\rstep {done}/{steps}  {time.perf_counter() - t0:.1f} s", end="", file=sys.stderr)

    np.save(args.out, traj)
    print(f"\nwrote {args.out}  shape ({n_ic}, {steps + 1}, 8)", file=sys.stderr)


if __name__ == "__main__":
    main()
