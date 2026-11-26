"""
Critical value of the nonlocal coefficient alpha.

Linear homogeneous problem:

    u_t = u_xx,   u(1, t) = 0,   u(0, t) = alpha * int_0^1 u(x, t) dx.

Separated solutions u = exp(sigma t) phi(x), phi'' = sigma phi, phi(1) = 0:

  * sigma = -mu^2 < 0 :  phi = sin(mu (1 - x)).  The nonlocal condition reads
        sin(mu) = alpha (1 - cos(mu)) / mu
        <=>  2 sin(mu/2) [ cos(mu/2) - (alpha/mu) sin(mu/2) ] = 0,
    i.e. either mu = 2 n pi (eigenvalues -4 n^2 pi^2, independent of alpha),
    or  mu cot(mu/2) = alpha.
  * sigma = 0 :  phi = 1 - x, phi(0) = 1, int phi = 1/2  =>  alpha = 2.
  * sigma = kappa^2 > 0 :  phi = sinh(kappa (1 - x)),
        sinh(kappa) = alpha (cosh(kappa) - 1) / kappa
        <=>  alpha = kappa coth(kappa/2).

kappa coth(kappa/2) increases strictly from 2 (kappa -> 0) to infinity, so a
growing mode exists if and only if alpha > 2, and it is unique.

With z = mu/2 the alpha-dependent branch is  z cot z = alpha/2, which is the
eigen-condition of the SELF-ADJOINT Robin problem
    -psi'' = z^2 psi on (0, 1),  psi(0) = 0,  psi'(1) = (alpha/2) psi(1).
Its eigenvalues z^2 are therefore real: the spectrum of the nonlocal problem
is real, and alpha* = 2 separates decay from exponential growth.

Nothing here is hard-coded: alpha* is obtained symbolically (SymPy) and
numerically (spectrum of the finite-difference operator + bisection).

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import numpy as np
import sympy as sp
from scipy.optimize import brentq


# ===================================================================
# 1. Symbolic derivation
# ===================================================================

def symbolic_derivation():
    """Derive the critical value with SymPy.  Returns a dict of results."""
    x = sp.symbols("x", real=True)
    kappa, mu, alpha = sp.symbols("kappa mu alpha", positive=True)

    # --- sigma = 0 : phi = 1 - x -------------------------------------
    phi0 = 1 - x
    alpha_zero_mode = sp.solve(
        sp.Eq(phi0.subs(x, 0), alpha * sp.integrate(phi0, (x, 0, 1))), alpha)

    # --- sigma = kappa^2 : phi = sinh(kappa (1 - x)) ------------------
    phi_g = sp.sinh(kappa * (1 - x))
    assert sp.simplify(sp.diff(phi_g, x, 2) - kappa ** 2 * phi_g) == 0
    assert phi_g.subs(x, 1) == 0
    alpha_growing = sp.solve(
        sp.Eq(phi_g.subs(x, 0), alpha * sp.integrate(phi_g, (x, 0, 1))),
        alpha)[0]
    closed_form = kappa * sp.coth(kappa / 2)
    identity_residual = sp.simplify(
        (alpha_growing - closed_form).rewrite(sp.exp))

    # limit kappa -> 0 and small-kappa expansion
    alpha_limit = sp.limit(closed_form, kappa, 0, "+")
    series = sp.series(closed_form, kappa, 0, 6).removeO()

    # monotonicity: d/dkappa = (sinh kappa - kappa) / (2 sinh^2(kappa/2)) > 0
    derivative = sp.diff(closed_form, kappa)
    derivative_closed = (sp.sinh(kappa) - kappa) / (2 * sp.sinh(kappa / 2) ** 2)
    derivative_residual = sp.simplify(
        (derivative - derivative_closed).rewrite(sp.exp))

    # --- sigma = -mu^2 : phi = sin(mu (1 - x)) ------------------------
    phi_d = sp.sin(mu * (1 - x))
    assert sp.simplify(sp.diff(phi_d, x, 2) + mu ** 2 * phi_d) == 0
    charac = phi_d.subs(x, 0) - alpha * sp.integrate(phi_d, (x, 0, 1))
    factorised = 2 * sp.sin(mu / 2) * (sp.cos(mu / 2)
                                       - alpha / mu * sp.sin(mu / 2))
    factor_residual = sp.simplify(sp.expand_trig(charac - factorised))

    return {
        "alpha_zero_mode": alpha_zero_mode,          # [2]
        "alpha_growing": alpha_growing,              # kappa sinh/(cosh - 1)
        "identity_residual": identity_residual,      # 0
        "alpha_limit": alpha_limit,                  # 2
        "series": series,                            # 2 + kappa^2/6 - ...
        "derivative_residual": derivative_residual,  # 0
        "factor_residual": factor_residual,          # 0
    }


# ===================================================================
# 2. Characteristic equation (continuous problem)
# ===================================================================

def alpha_of_kappa(kappa):
    """alpha for which exp(kappa^2 t) sinh(kappa (1 - x)) is a solution."""
    kappa = np.asarray(kappa, dtype=float)
    return kappa / np.tanh(kappa / 2.0)


def alpha_of_mu(mu):
    """alpha for which exp(-mu^2 t) sin(mu (1 - x)) is a solution."""
    mu = np.asarray(mu, dtype=float)
    return mu / np.tan(mu / 2.0)


def leading_eigenvalue(alpha):
    """Largest eigenvalue sigma_1(alpha) of the continuous problem.

    alpha > 2 : sigma_1 = +kappa^2, kappa coth(kappa/2) = alpha
    alpha = 2 : sigma_1 = 0
    alpha < 2 : sigma_1 = -mu^2,   mu cot(mu/2) = alpha, mu in (0, 2 pi)
    """
    if abs(alpha - 2.0) < 1e-14:
        return 0.0
    if alpha > 2.0:
        kappa = brentq(lambda k: alpha_of_kappa(k) - alpha, 1e-8, 10.0 * alpha)
        return kappa ** 2
    mu = brentq(lambda m: alpha_of_mu(m) - alpha, 1e-8, 2.0 * np.pi - 1e-10)
    return -mu ** 2


# ===================================================================
# 3. Spectrum of the finite-difference operator
# ===================================================================

def fd_operator(alpha, N=200):
    """Second-difference operator on u_1..u_{N-1} with the nonlocal condition
    (trapezoidal rule) eliminated:  u_0 = alpha h sum_i u_i / (1 - alpha h/2).
    """
    h = 1.0 / N
    n = N - 1
    L = (np.diag(-2.0 * np.ones(n)) + np.diag(np.ones(n - 1), 1)
         + np.diag(np.ones(n - 1), -1)) / h ** 2
    L[0, :] += alpha * h / (1.0 - alpha * h / 2.0) / h ** 2
    return L


def discrete_spectrum(alpha, N=200):
    """Eigenvalues of the semi-discrete problem du/dt = L u."""
    return np.linalg.eigvals(fd_operator(alpha, N))


def discrete_leading_eigenvalue(alpha, N=200):
    eig = discrete_spectrum(alpha, N)
    return eig[np.argmax(eig.real)]


def critical_alpha_numerical(N=200, lo=0.5, hi=4.0):
    """alpha at which the leading discrete eigenvalue crosses zero."""
    return brentq(lambda a: discrete_leading_eigenvalue(a, N).real, lo, hi,
                  xtol=1e-12)


# ===================================================================
# Report
# ===================================================================

def report(N=200):
    sym = symbolic_derivation()
    print("=== Symbolic derivation ===")
    print(f"  zero mode phi = 1 - x requires alpha = {sym['alpha_zero_mode']}")
    print(f"  growing mode: alpha = {sym['alpha_growing']}")
    print(f"    minus kappa*coth(kappa/2)       -> {sym['identity_residual']}")
    print(f"    limit kappa -> 0+               -> {sym['alpha_limit']}")
    print(f"    series                          -> {sym['series']}")
    print(f"    d/dkappa - (sinh k - k)/(2 sinh^2(k/2)) -> "
          f"{sym['derivative_residual']}")
    print(f"  decaying branch factorisation residual -> {sym['factor_residual']}")

    print(f"\n=== Leading eigenvalue: continuous vs finite differences (N={N}) ===")
    print("   alpha    sigma_1 (charac. eq.)   sigma_1 (FD)    max|Im|")
    for a in [0.5, 1.0, 1.5, 1.9, 1.99, 2.0, 2.01, 2.5, 3.0]:
        eig = discrete_spectrum(a, N)
        lead = eig[np.argmax(eig.real)]
        print(f"  {a:6.2f}   {leading_eigenvalue(a):+18.6f}   "
              f"{lead.real:+14.6f}   {np.max(np.abs(eig.imag)):.1e}")

    print("\n=== Critical alpha from the discrete spectrum (bisection) ===")
    for n in [50, 100, 200, 400]:
        a_star = critical_alpha_numerical(n)
        print(f"  N = {n:4d}:  alpha* = {a_star:.10f}   |alpha* - 2| = "
              f"{abs(a_star - 2):.2e}")


if __name__ == "__main__":
    report()
