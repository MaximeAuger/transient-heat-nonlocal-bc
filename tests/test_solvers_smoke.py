"""Short end-to-end runs of both solvers (small budget)."""
import numpy as np
import pytest
import torch

from exact_solution import DTYPE, manufactured, source_free
from fd_reference import solve_crank_nicolson
from network import HardConstrainedNet
from pinn_solver import train_pinn
from vpinn_solver import train_vpinn

SMALL = {"n_adam": 300, "n_lbfgs": 300, "n_x": 16, "n_t": 16,
         "n_hidden": 16, "n_layers": 3}


@pytest.mark.parametrize("kind,alpha", [("decay", 1.0), ("oscillating", 1.5)])
def test_hard_constraints_are_exact(kind, alpha):
    prob = manufactured(kind, alpha=alpha, beta=1.0)
    torch.manual_seed(0)
    model = HardConstrainedNet(prob).to(DTYPE)
    s = torch.linspace(0, 1, 11, dtype=DTYPE).reshape(-1, 1)
    u_init = model(s, torch.zeros_like(s)).detach().numpy().ravel()
    assert np.allclose(u_init, prob.u0(s.numpy().ravel()), atol=1e-14)
    u_right = model(torch.ones_like(s), s).detach().numpy().ravel()
    assert np.allclose(u_right, prob.g1(s.numpy().ravel()), atol=1e-14)


@pytest.mark.parametrize("train", [train_pinn, train_vpinn])
@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_training_reduces_loss_and_error(train, beta):
    res = train({**SMALL, "beta": beta}, verbose=False)
    loss = res["history"]["loss"]
    assert np.isfinite(loss[-1]) and loss[-1] < 1e-2 * loss[0]
    assert res["errors"]["L2_rel"] < 2e-2
    assert res["errors"]["nonlocal_max"] < 2e-2
    assert not res["lbfgs_failed"]


def test_training_is_reproducible():
    a = train_vpinn({**SMALL, "n_lbfgs": 40}, verbose=False)
    b = train_vpinn({**SMALL, "n_lbfgs": 40}, verbose=False)
    assert a["errors"]["L2"] == b["errors"]["L2"]


def test_lambda_matches_exact_flux():
    res = train_vpinn({**SMALL, "n_adam": 600, "n_lbfgs": 600}, verbose=False)
    assert res["errors"]["flux_lam_rel"] < 5e-2
    assert len(res["lam_coeffs"]) == res["config"]["n_test_t"]


def test_source_free_problem_against_fd_reference():
    cfg = {**SMALL, "problem": "source_free", "alpha": 1.0}
    ref = solve_crank_nicolson(source_free(1.0), 200, 200)
    res = train_pinn(cfg, verbose=False, reference=ref)
    assert res["errors"]["L2_rel"] < 5e-2
