"""Convergence of the Crank-Nicolson finite-difference reference."""
import numpy as np
import pytest

from exact_solution import manufactured
from fd_reference import fd_errors, solve_crank_nicolson


@pytest.mark.parametrize("kind,alpha,beta", [
    ("decay", 1.0, 0.0), ("decay", 1.0, 2.0), ("oscillating", 1.0, 1.0),
    ("decay", 0.5, 0.0),
])
def test_second_order_convergence(kind, alpha, beta):
    prob = manufactured(kind, alpha, beta)
    e = [fd_errors(prob, n, n)["L2"] for n in (40, 80, 160)]
    orders = np.log2(np.array(e[:-1]) / np.array(e[1:]))
    assert np.all(orders > 1.9) and np.all(orders < 2.1), orders
    assert e[-1] < 1e-4


def test_converges_close_to_the_critical_alpha():
    """alpha = 1.99: a nearly neutral mode delays the asymptotic regime."""
    prob = manufactured("decay", alpha=1.99)
    e = [fd_errors(prob, n, n)["L2"] for n in (40, 80, 160, 320)]
    orders = np.log2(np.array(e[:-1]) / np.array(e[1:]))
    assert np.all(orders > 1.8) and np.all(np.diff(orders) > 0), orders


def test_converges_in_the_growing_mode_regime():
    """alpha = 3 > alpha*: errors are amplified but the scheme converges."""
    prob = manufactured("decay", alpha=3.0)
    e = [fd_errors(prob, n, n)["L2"] for n in (100, 200, 400)]
    orders = np.log2(np.array(e[:-1]) / np.array(e[1:]))
    assert np.all(orders > 1.8), orders
    stable = fd_errors(manufactured("decay", alpha=1.0), 400, 400)["L2"]
    assert e[-1] > 50 * stable          # amplification by the growing mode


def test_separate_orders_in_space_and_time():
    prob = manufactured("decay", 1.0, 0.0)
    es = [fd_errors(prob, n, 2000)["L2"] for n in (20, 40)]
    et = [fd_errors(prob, 2000, m)["L2"] for m in (5, 10)]
    assert 1.9 < np.log2(es[0] / es[1]) < 2.1
    assert 1.9 < np.log2(et[0] / et[1]) < 2.1


def test_discrete_nonlocal_condition_and_flux():
    prob = manufactured("decay", 1.5, 1.0)
    sol = solve_crank_nicolson(prob, 200, 200)
    viol = sol["u"][0] - prob.alpha * sol["integral"] - prob.g0(sol["t"])
    assert np.max(np.abs(viol[1:])) < 1e-9
    assert np.max(np.abs(sol["u"][-1] - prob.g1(sol["t"]))) < 1e-14
    rel = np.max(np.abs(sol["flux0"] - prob.flux0(sol["t"]))) \
        / np.max(np.abs(prob.flux0(sol["t"])))
    assert rel < 5e-3
