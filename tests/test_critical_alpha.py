"""alpha* = 2 separates decay from exponential growth (derived, not assumed)."""
import numpy as np
import pytest
import sympy as sp

from critical_alpha import (alpha_of_kappa, alpha_of_mu,
                            critical_alpha_numerical, discrete_spectrum,
                            leading_eigenvalue, symbolic_derivation)
from exact_solution import source_free
from fd_reference import solve_crank_nicolson


def test_symbolic_derivation():
    sym = symbolic_derivation()
    assert sym["alpha_zero_mode"] == [2]
    assert sym["identity_residual"] == 0      # alpha = kappa coth(kappa/2)
    assert sym["alpha_limit"] == 2            # kappa -> 0+
    assert sym["derivative_residual"] == 0    # (sinh k - k) / (2 sinh^2(k/2))
    assert sym["factor_residual"] == 0
    kappa = sp.symbols("kappa", positive=True)
    expected = 2 + kappa ** 2 / 6 - kappa ** 4 / 360
    assert sp.simplify(sym["series"] - expected) == 0


def test_growing_branch_is_increasing_and_above_two():
    kappa = np.linspace(1e-3, 20.0, 4000)
    a = alpha_of_kappa(kappa)
    assert np.all(a > 2.0) and np.all(np.diff(a) > 0.0)
    # sinh(kappa) - kappa > 0: sign of the derivative
    assert np.all(np.sinh(kappa) - kappa > 0.0)


def test_decaying_branch_is_decreasing_and_below_two():
    mu = np.linspace(1e-3, 2 * np.pi - 1e-3, 4000)
    a = alpha_of_mu(mu)
    assert np.all(a < 2.0) and np.all(np.diff(a) < 0.0)


@pytest.mark.parametrize("N", [50, 100, 200])
def test_critical_alpha_from_discrete_spectrum(N):
    assert abs(critical_alpha_numerical(N) - 2.0) < 1e-8


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 1.9, 1.99, 2.01, 2.5, 3.0])
def test_discrete_spectrum_matches_characteristic_equation(alpha):
    eig = discrete_spectrum(alpha, 200)
    assert np.max(np.abs(eig.imag)) < 1e-8          # real spectrum
    lead = np.max(eig.real)
    exact = leading_eigenvalue(alpha)
    assert abs(lead - exact) < 2e-3 * max(1.0, abs(exact))
    assert (lead > 0) == (alpha > 2.0)
    # exactly one unstable eigenvalue above the critical value
    assert np.sum(eig.real > 0) == (1 if alpha > 2.0 else 0)


@pytest.mark.parametrize("alpha", [1.0, 2.5, 3.0])
def test_time_integration_reproduces_the_growth_rate(alpha):
    """Source-free problem integrated by Crank-Nicolson: the late-time rate
    d/dt log|u| equals the leading eigenvalue (negative below alpha* = 2,
    positive above)."""
    sol = solve_crank_nicolson(source_free(alpha, T=2.0), 200, 800)
    amp = np.abs(sol["u"][0])                    # u(0, t)
    t = sol["t"]
    sel = t > 1.2
    rate = np.polyfit(t[sel], np.log(amp[sel]), 1)[0]
    exact = leading_eigenvalue(alpha)
    assert abs(rate - exact) < 2e-2 * abs(exact)
    assert (rate > 0) == (alpha > 2.0)
