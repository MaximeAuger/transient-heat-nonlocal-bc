"""
Finite-difference reference: Crank-Nicolson in time, centred differences in
space, nonlocal condition assembled with the trapezoidal rule.

    u_t - [k(u) u_x]_x = f,   k(u) = 1 + beta u^2
    u(0, t) = alpha * int_0^1 u dx + g0(t),   u(1, t) = g1(t),   u(x, 0) = u0

Unknowns at each time level: u_0, ..., u_{N-1}  (u_N = g1 is eliminated).

    row 0      : u_0 - alpha * trap(u) - g0(t^{n+1}) = 0
    rows 1..N-1: (u_i^{n+1} - u_i^n)/dt
                   = 1/2 [A(u^{n+1})_i + A(u^n)_i] + 1/2 [f_i^{n+1} + f_i^n]

    A(u)_i = [ k_{i+1/2} (u_{i+1} - u_i) - k_{i-1/2} (u_i - u_{i-1}) ] / h^2,
    k_{i+1/2} = k( (u_i + u_{i+1}) / 2 ).

The nonlinear system (beta > 0) is solved by Newton-Raphson at every step;
for beta = 0 the matrix is constant and factorised once.  The scheme is
second-order accurate in both h and dt.

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu

from exact_solution import k_conductivity, dk_conductivity


def _spatial_operator(u, beta, h):
    """A(u)_i for i = 1..N-1 and its tridiagonal Jacobian entries."""
    um = 0.5 * (u[:-1] + u[1:])                 # midpoints, size N
    k = k_conductivity(um, beta)
    dk = dk_conductivity(um, beta)
    du = u[1:] - u[:-1]                         # size N
    A = (k[1:] * du[1:] - k[:-1] * du[:-1]) / h ** 2
    lower = (k[:-1] - 0.5 * dk[:-1] * du[:-1]) / h ** 2          # dA_i/du_{i-1}
    diag = ((0.5 * dk[1:] * du[1:] - k[1:])
            - (0.5 * dk[:-1] * du[:-1] + k[:-1])) / h ** 2       # dA_i/du_i
    upper = (0.5 * dk[1:] * du[1:] + k[1:]) / h ** 2             # dA_i/du_{i+1}
    return A, lower, diag, upper


def _assemble(lower, diag, upper, alpha, h, dt, N):
    """Jacobian of the Crank-Nicolson residual w.r.t. u_0..u_{N-1}."""
    rows, cols, vals = [], [], []
    # row 0: nonlocal condition (trapezoidal rule), dense
    rows.append(np.zeros(N, dtype=int))
    cols.append(np.arange(N))
    r0 = -alpha * h * np.ones(N)
    r0[0] = 1.0 - alpha * h / 2.0
    vals.append(r0)
    i = np.arange(1, N)
    # sub-diagonal (column i-1)
    rows.append(i); cols.append(i - 1); vals.append(-0.5 * lower)
    # diagonal
    rows.append(i); cols.append(i); vals.append(1.0 / dt - 0.5 * diag)
    # super-diagonal (column i+1 < N)
    rows.append(i[:-1]); cols.append(i[:-1] + 1); vals.append(-0.5 * upper[:-1])
    return sps.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N))


def solve_crank_nicolson(problem, N=400, M=400, tol=1e-11, max_newton=50,
                         verbose=False):
    """Solve the problem on a uniform (N+1) x (M+1) space-time grid.

    Returns
    -------
    dict with
        x (N+1,), t (M+1,), u (N+1, M+1),
        flux0 (M+1,)   : -k(u) u_x at x = 0 (one-sided, second order),
        integral (M+1,): trapezoidal int_0^1 u dx,
        newton_iters   : maximum Newton iterations used in a step.
    """
    alpha, beta, T = problem.alpha, problem.beta, problem.T
    h, dt = 1.0 / N, T / M
    x = np.linspace(0.0, 1.0, N + 1)
    t = np.linspace(0.0, T, M + 1)
    g0, g1 = problem.g0(t), problem.g1(t)

    U = np.zeros((N + 1, M + 1))
    U[:, 0] = problem.u0(x)
    linear = (beta == 0.0)
    lu = None
    worst = 0

    f_old = problem.f(x, np.full_like(x, t[0]))
    for n in range(M):
        u_old = U[:, n]
        f_new = problem.f(x, np.full_like(x, t[n + 1]))
        A_old, lo, di, up = _spatial_operator(u_old, beta, h)
        rhs_explicit = u_old[1:N] / dt + 0.5 * A_old + 0.5 * (f_old[1:N] + f_new[1:N])

        u = u_old.copy()
        u[N] = g1[n + 1]
        for it in range(max_newton):
            A_new, lo, di, up = _spatial_operator(u, beta, h)
            F = np.empty(N)
            trap = h * (0.5 * u[0] + np.sum(u[1:N]) + 0.5 * u[N])
            F[0] = u[0] - alpha * trap - g0[n + 1]
            F[1:] = u[1:N] / dt - 0.5 * A_new - rhs_explicit
            res = np.max(np.abs(F))
            if it > 0 and res < tol * max(1.0, np.max(np.abs(u)) / dt):
                break
            if linear:
                if lu is None:
                    lu = splu(_assemble(lo, di, up, alpha, h, dt, N),
                              permc_spec="NATURAL")
                u[:N] -= lu.solve(F)
            else:
                J = _assemble(lo, di, up, alpha, h, dt, N)
                u[:N] -= splu(J, permc_spec="NATURAL").solve(F)
        else:
            print(f"  WARNING: Newton did not converge at step {n + 1} "
                  f"(residual {res:.2e})")
        worst = max(worst, it)
        U[:, n + 1] = u
        f_old = f_new
        if verbose and (n + 1) % max(1, M // 5) == 0:
            print(f"  CN step {n + 1:5d}/{M}  Newton its = {it}  "
                  f"max|u| = {np.max(np.abs(u)):.4e}")

    ux0 = (-3.0 * U[0] + 4.0 * U[1] - U[2]) / (2.0 * h)
    flux0 = -k_conductivity(U[0], beta) * ux0
    integral = h * (0.5 * U[0] + np.sum(U[1:N], axis=0) + 0.5 * U[N])
    return {"x": x, "t": t, "u": U, "flux0": flux0, "integral": integral,
            "newton_iters": worst}


def fd_errors(problem, N=400, M=400):
    """Errors of the Crank-Nicolson solution against a manufactured solution."""
    sol = solve_crank_nicolson(problem, N, M)
    xg, tg = np.meshgrid(sol["x"], sol["t"], indexing="ij")
    err = sol["u"] - problem.u(xg, tg)
    l2 = np.sqrt(np.trapezoid(np.trapezoid(err ** 2, sol["t"], axis=1), sol["x"]))
    return {"L2": l2, "Linf": np.max(np.abs(err)),
            "flux_Linf": np.max(np.abs(sol["flux0"] - problem.flux0(sol["t"]))),
            "sol": sol}


if __name__ == "__main__":
    from exact_solution import manufactured

    for kind, alpha, beta in [("decay", 1.0, 0.0), ("decay", 1.0, 2.0),
                              ("oscillating", 1.0, 1.0), ("decay", 3.0, 0.0)]:
        prob = manufactured(kind, alpha, beta)
        print(f"{prob}")
        prev = None
        for n in [25, 50, 100, 200]:
            e = fd_errors(prob, n, n)
            rate = "" if prev is None else f"   order = {np.log2(prev / e['L2']):.2f}"
            print(f"  N = M = {n:4d}:  L2 = {e['L2']:.3e}  Linf = {e['Linf']:.3e}"
                  f"  flux = {e['flux_Linf']:.3e}{rate}")
            prev = e["L2"]
