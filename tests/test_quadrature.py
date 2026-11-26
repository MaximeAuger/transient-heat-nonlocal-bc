"""Quadratures, test functions and consistency of both discrete losses."""
import numpy as np
import pytest
import torch

from exact_solution import DTYPE, manufactured
from utils import (SpaceTimeGrid, LegendreTestFunctions, gauss_legendre,
                   legendre_time_basis, nonlocal_residual)
import pinn_solver
import vpinn_solver


class ExactModel(torch.nn.Module):
    """Stand-in for the network that returns the manufactured solution."""

    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def forward(self, x, t):
        return self.problem.u_torch(x, t)


@pytest.mark.parametrize("n", [2, 5, 16, 32])
def test_gauss_legendre_is_exact_up_to_degree_2n_minus_1(n):
    x, w = gauss_legendre(n, 0.0, 2.0)
    assert abs(np.sum(w) - 2.0) < 1e-13
    for p in (1, n, 2 * n - 1):
        exact = 2.0 ** (p + 1) / (p + 1)
        assert abs(np.sum(w * x ** p) - exact) < 1e-11 * exact


def test_gauss_legendre_is_not_exact_beyond():
    x, w = gauss_legendre(3)
    assert abs(np.sum(w * x ** 6) - 1.0 / 7.0) > 1e-6


@pytest.mark.parametrize("norm", ["none", "h1", "l2"])
def test_space_test_functions(norm):
    tf = LegendreTestFunctions(10, normalise=norm)
    assert np.max(np.abs(tf.eval_v(np.array([1.0])))) < 1e-13     # v(1) = 0
    assert np.allclose(tf.eval_v(np.array([0.0]))[0], tf.v_at_zero())
    x = np.linspace(0.1, 0.9, 7)
    h = 1e-6
    fd = (tf.eval_v(x + h) - tf.eval_v(x - h)) / (2 * h)
    assert np.allclose(fd, tf.eval_dv(x), rtol=1e-6, atol=1e-6)


def test_h1_normalisation_gives_unit_seminorm():
    tf = LegendreTestFunctions(12, normalise="h1")
    x, w = gauss_legendre(40)
    assert np.allclose(w @ tf.eval_dv(x) ** 2, 1.0)


def test_time_basis_is_orthonormal_when_normalised():
    T = 5.0
    t, w = gauss_legendre(40, 0.0, T)
    B = legendre_time_basis(8, t, T, normalise=True)
    assert np.allclose(B.T @ (w[:, None] * B), np.eye(8), atol=1e-12)


@pytest.mark.parametrize("alpha", [1.0, 1.99, 3.0])
def test_nonlocal_residual_vanishes_for_exact_solution(alpha):
    prob = manufactured("oscillating", alpha=alpha)
    grid = SpaceTimeGrid(prob, 24, 16)
    model = ExactModel(prob)
    u = grid.to_grid(model(grid.x, grid.t))
    assert torch.max(torch.abs(nonlocal_residual(model, prob, grid, u))) < 1e-12


@pytest.mark.parametrize("kind,beta", [("decay", 0.0), ("decay", 2.0),
                                       ("oscillating", 1.0)])
def test_pinn_loss_vanishes_for_exact_solution(kind, beta):
    prob = manufactured(kind, beta=beta)
    grid = SpaceTimeGrid(prob, 16, 16)
    loss, parts = pinn_solver._loss_pinn(
        ExactModel(prob), prob, grid, pinn_solver.DEFAULT_CONFIG)
    assert parts["pde"] < 1e-20 and parts["I"] < 1e-24


@pytest.mark.parametrize("kind,beta,norm", [("decay", 0.0, "h1"),
                                            ("decay", 2.0, "none"),
                                            ("oscillating", 1.0, "l2")])
def test_weak_residual_vanishes_for_exact_solution_and_flux(kind, beta, norm):
    """Checks the weak form, its sign conventions and the role of lambda(t):
    with u exact and lambda = L2 projection of -k(u) u_x(0, t) on the time
    test space, every weak residual R_ij must vanish."""
    prob = manufactured(kind, beta=beta)
    cfg = {**vpinn_solver.DEFAULT_CONFIG, "n_x": 40, "n_t": 40,
           "n_test_x": 10, "n_test_t": 6, "normalise_tests": norm}
    grid = SpaceTimeGrid(prob, cfg["n_x"], cfg["n_t"])
    vd = vpinn_solver.VPINNData(prob, grid, cfg)
    # Legendre coefficients of the exact flux: c_m = (2m+1)/T int lam P_m
    P = legendre_time_basis(cfg["n_test_t"], grid.tq, prob.T)
    c = (2 * np.arange(cfg["n_test_t"]) + 1) / prob.T \
        * (P.T @ (grid.wt_np * prob.flux0(grid.tq)))
    loss, parts = vpinn_solver._loss_vpinn(
        ExactModel(prob), torch.tensor(c, dtype=DTYPE), prob, grid, vd, cfg)
    assert parts["weak"] < 1e-20

    # ... and it must NOT vanish with a wrong flux (lambda = 0)
    loss0, parts0 = vpinn_solver._loss_vpinn(
        ExactModel(prob), torch.zeros(len(c), dtype=DTYPE), prob, grid, vd, cfg)
    assert parts0["weak"] > 1e-4
