import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit



# Kerr metric set up
M = 1.0
a = 0.0



# Inverse Kerr metric (uses the inverse metric like the Fantasty code). Takes in
#a vector of length 4 representing the 4 Boyder-Lindquist coordinates: t, r,
#theta, and phi
def inverse_metric(x):

    t = x[0]
    r = x[1]
    theta = x[2]
    phi = x[3]

    #Making it easier and more compact to write out the full Kerr metric
    Sigma = r**2 + a**2 * np.cos(theta)**2
    Delta = r**2 - 2*M*r + a**2

    #Set up a 4x4 matrix of zeroes
    g = np.zeros((4, 4))

    #Input all the spacetime metric components (Kerr for this example)
    g[0, 0] = -((r**2 + a**2)**2 - a**2 * Delta * np.sin(theta)**2) / (Delta * Sigma)
    g[0, 1] = 0.0
    g[0, 2] = 0.0
    g[0, 3] = -2 * M * a * r / (Delta * Sigma)

    g[1, 0] = g[0, 1]
    g[1, 1] = Delta / Sigma
    g[1, 2] = 0.0
    g[1, 3] = 0.0

    g[2, 0] =  g[0, 2]
    g[2, 1] =  g[1, 2]
    g[2, 2] = 1.0 / Sigma
    g[2, 3] = 0.0

    g[3, 0] = g[0, 3]
    g[3, 1] = g[1, 3]
    g[3, 2] = g[2, 3]
    g[3, 3] = (Delta - a**2 * np.sin(theta)**2) / (Delta * Sigma * np.sin(theta)**2)

    return g



# Numerical derivatives of inverse metric
def dgdx(x, delta=1e-6):
    # Find the number of variables we're working with and create an n by n by n array
    #that will hold all the partial derivvatives of every entry of the inverse metric
    #matrix
    n = len(x)
    dg = np.zeros((n, n, n))

    # Uses central finite differences to approximate each partial derivative.
    #The n-th loop computes all 4 partial derivatives for the n-th spacetime
    #coordinate.
    for w in range(n):
        dx = np.zeros(n)
        dx[w] = delta
        g_plus = inverse_metric(x + dx)
        g_minus = inverse_metric(x - dx)
        dg[w] = (g_plus - g_minus) / (2 * delta)

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
    dx = np.dot(g,p)

    dp = np.zeros(n)

    # Uses a triple for-loop to sum over the 3 indices in the equation for the
    #momentum derivative
    for w in range(n):
        for u in range(n):
            for v in range(n):
                dp[w] = dp[w] - 0.5 * dg[w, u, v] * p[u] * p[v]

    return np.concatenate([dx, dp])



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
    for _ in range(max_iter):
        k1_new = f(y + h * (r4a11 * k1 + r4a12 * k2))
        k2_new = f(y + h * (r4a21 * k1 + r4a22 * k2))

        # Checks if the difference (in the regular 2-norm) of k1 and the updated
        #k1 is less than the given tolerance, and if the same is true for k2 and
        #the updated k2.
        if ((np.linalg.norm(k1_new - k1) < tol) and (np.linalg.norm(k2_new - k2) < tol)):
            break

        # Replaces the previous guess with the new one
        k1 = k1_new
        k2 = k2_new

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
    for _ in range(max_iter):
        k1_new = f(y + h * (r6a11*k1 + r6a12*k2 + r6a13*k3))
        k2_new = f(y + h * (r6a21*k1 + r6a22*k2 + r6a23*k3))
        k3_new = f(y + h * (r6a31*k1 + r6a32*k2 + r6a33*k3))

        # Convergence check (2-norm)
        if (np.linalg.norm(k1_new - k1) < tol and
            np.linalg.norm(k2_new - k2) < tol and
            np.linalg.norm(k3_new - k3) < tol):
            break

        # Replaces the previous guess with the new one
        k1 = k1_new
        k2 = k2_new
        k3 = k3_new

    # Uses the final guesses and the symplectic rk6 coefficients to update y.
    return y + h * (r6b1*k1 + r6b2*k2 + r6b3*k3)



# We'll compute an rk4 and an rk6 step, find the error between them, and decide
#whether to accept the rk4 step, or shrink the step size and try again until the
#error is less than a given tolerance.
def symplectic_rk46_comparison(f, y, h, tol=0.1, max_shrinks=1, rk_tol=5e-11, rk_max_iter=20):

    # Define the current default step size
    h_current = h

    # A for loop = computes an rk4 and rk6 step until the error between them is
    #below a given tolerance, then computes them again with a smaller step size
    #if it's not
    for _ in range(max_shrinks + 1):

        # Compute a symplectic rk4 step
        y_rk4 = symplectic_rk4(f, y, h_current, tol=rk_tol, max_iter=rk_max_iter)

        # Compute a symplectic rk6 step
        y_rk6 = symplectic_rk6(f, y, h_current, tol=rk_tol, max_iter=rk_max_iter)

        # Find the 2 norm of their difference
        error = np.linalg.norm(y_rk6 - y_rk4)

        # Accept symplectic rk4 step if error is below the tolerance
        if error < tol:
            return y_rk4

        # Update step size using formula that's used in the rkf method
        h_current = 0.9 * h_current * (tol / error)**(1/5)

    # If the error still isn't below the tolerance after max_shrinks, accept the rk4 step
    return y_rk4





# Integrate Geodesic
def integrate_geodesic(y0, h, steps):
    # Makes a copy of the initial conditions vector and appends it to a trajectory
    #list.
    y = y0.copy()
    geodesic = []
    geodesic.append(y.copy())

    # Compute the position/momentum vector using the symplectic rk4, adds it to the
    #trajectory list, and then computes the next position/momentum in the geodesic
    for _ in range(steps):
        y = symplectic_rk46_comparison(hamiltonian, y, h)
        geodesic.append(y.copy())

    # Returns the list of positions/momenta for the geodesic
    return np.array(geodesic)



# Graph the geodesic trajectory
def graph_geodesic(geodesic):
    # Singling out each component of the trajectory as their own vectors
    r = geodesic[:, 1]
    theta = geodesic[:, 2]
    phi = geodesic[:, 3]

    # Coordinate transformations between spherical and cartesian
    X = r * np.sin(theta) * np.cos(phi)
    Y = r * np.sin(theta) * np.sin(phi)
    Z = r * np.cos(theta)

    # Setting up the graph labels and title
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.plot(X, Y, Z, 'r', linewidth=1.0, color = 'purple')
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Schwarzschild Geodesic")
    plt.show()



# Initial Conditions

# Initial position
t0 = 0.0
r0 = 15.0
theta0 = np.pi / 2
phi0 = 0.0

# Initial momenta
p_t0 = -0.9764550153430405
p_r0 = 0.0
p_theta0 = 3.8
p_phi0 = 3.0

# Puts initial conditions into a single vector
y0 = np.array([t0, r0, theta0, phi0, p_t0, p_r0, p_theta0, p_phi0])

# Calculates and graphs the geodesic for the given initial conditions.
geodesic = integrate_geodesic(y0, 1, 15000)
graph_geodesic(geodesic)
