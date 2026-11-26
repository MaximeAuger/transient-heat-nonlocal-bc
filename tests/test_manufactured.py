"""The manufactured solutions satisfy the PDE and all conditions."""
import numpy as np
import pytest
import sympy as sp

from exact_solution import X, TT, manufactured, source_free, verify

CASES = [(kind, alpha, beta)
         for kind in ("decay", "oscillating")
         for alpha, beta in [(1.0, 0.0), (1.0, 1.0), (0.5, 2.0), (1.99, 0.0),
                             (2.5, 0.5), (3.0, 5.0)]]


@pytest.mark.parametrize("kind,alpha,beta", CASES)
def test_residual_and_conditions_vanish(kind, alpha, beta):
    out = verify(manufactured(kind, alpha, beta), verbose=False)
    assert out["pde"] < 1e-12          # relative PDE residual (autograd)
    assert out["nonlocal"] < 1e-12
    assert out["dirichlet"] < 1e-12
    assert out["initial"] < 1e-12
    assert out["integral"] < 1e-12


@pytest.mark.parametrize("kind,alpha,beta", CASES)
def test_symbolic_residual_is_zero(kind, alpha, beta):
    """Independent symbolic check at a few rational points."""
    prob = manufactured(kind, alpha, beta)
    u, f = prob.sym["u"], prob.sym["f"]
    k = 1 + sp.Rational(str(beta)) * u ** 2
    res = sp.diff(u, TT) - sp.diff(k * sp.diff(u, X), X) - f
    for xv, tv in [(sp.Rational(1, 3), sp.Rational(1, 7)),
                   (sp.Rational(4, 5), sp.Rational(9, 10))]:
        assert abs(sp.N(res.subs({X: xv, TT: tv}), 30)) < 1e-20


@pytest.mark.parametrize("kind", ["decay", "oscillating"])
def test_alpha_one_gives_homogeneous_data(kind):
    prob = manufactured(kind, alpha=1.0, beta=0.0)
    assert prob.sym["g0"] == 0
    assert prob.sym["g1"] == 0


def test_oscillating_solution_is_not_monotone_in_time():
    prob = manufactured("oscillating")
    t = np.linspace(0.0, 1.0, 201)
    du = np.diff(prob.u(np.full_like(t, 0.3), t))
    assert np.any(du > 0) and np.any(du < 0)


def test_final_time_is_respected():
    prob = manufactured("decay", T=5.0)
    assert prob.T == 5.0
    assert verify(prob, verbose=False)["ok"]


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.5, 3.0])
def test_source_free_initial_state_is_compatible(alpha):
    prob = source_free(alpha)
    xi, wi = np.polynomial.legendre.leggauss(40)
    xq, wq = 0.5 * (xi + 1.0), 0.5 * wi
    u0_left = prob.u0(np.array([0.0]))[0]
    assert abs(u0_left - alpha * np.sum(wq * prob.u0(xq))) < 1e-12
    assert abs(prob.u0(np.array([1.0]))[0]) < 1e-14
    assert not prob.has_exact


def test_source_free_is_singular_at_critical_alpha():
    with pytest.raises(ValueError):
        source_free(2.0)


@pytest.mark.parametrize("gamma", [-1.0, 0.5, 2.0])
def test_nonlocal_initial_data(gamma):
    """phi = u(x, 0) - gamma u(x, T) for the Sapagovas-type condition (E5)."""
    prob = manufactured("decay", gamma=gamma)
    x = np.linspace(0.0, 1.0, 21)
    expected = prob.u(x, np.zeros_like(x)) - gamma * prob.u(x, np.full_like(x, prob.T))
    assert np.allclose(prob.ic_phi(x), expected, atol=1e-14)
    assert verify(prob, verbose=False)["ok"]      # PDE data are unchanged
