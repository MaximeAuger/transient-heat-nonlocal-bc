"""
PINN solver for the transient heat equation with a nonlocal integral
boundary condition (strong form).

PDE:  u_t - [k(u) u_x]_x = f(x, t),   k(u) = 1 + beta * u^2
BCs:  u(0, t) = alpha * int_0^1 u dx + g0(t),   u(1, t) = g1(t)
IC:   u(x, 0) = u0(x)

The strong residual is evaluated at space-time collocation points by
automatic differentiation on the flux, as in the steady-state code:

    flux = k(u_theta) * d/dx u_theta
    r    = d/dt u_theta - d/dx flux - f

The initial condition and the Dirichlet condition are imposed exactly by the
network ansatz (network.py).  The nonlocal condition is a penalty evaluated
at the Gauss-Legendre time nodes, with the spatial integral computed by the
Gauss-Legendre rule in space -- exactly the same term as in the VPINN.

Loss:
    L = w_f * mean_Q(r^2) + w_i * mean_t(r_I^2)

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import numpy as np
import torch

from exact_solution import make_problem, k_conductivity
from network import HardConstrainedNet
from utils import (
    COMMON_CONFIG, DEVICE, SpaceTimeGrid, nonlocal_loss, diff, optimise,
    compute_errors,
)
from exact_solution import DTYPE


# ===================================================================
# Default configuration
# ===================================================================

DEFAULT_CONFIG = {
    **COMMON_CONFIG,
    "w_f": 1.0,            # PDE residual weight
}


# ===================================================================
# Loss function
# ===================================================================

def _loss_pinn(model, problem, grid, cfg):
    """Total PINN loss (strong-form residual + nonlocal penalty)."""
    beta = problem.beta
    x = grid.x.clone().requires_grad_(True)
    t = grid.t.clone().requires_grad_(True)

    # --- PDE residual via flux differentiation ---
    u = model(x, t)
    u_x, u_t = torch.autograd.grad(
        u, [x, t], grad_outputs=torch.ones_like(u), create_graph=True)
    flux = k_conductivity(u, beta) * u_x                 # k(u) u_x
    flux_x = diff(flux, x)                               # [k(u) u_x]_x
    r = grid.to_grid(u_t - flux_x) - grid.f              # (n_x, n_t)
    l_pde = torch.sum(grid.wx * r ** 2 * grid.wt) / grid.T

    # --- Nonlocal condition: u(0,t) = alpha * int u dx + g0(t) ---
    l_I = nonlocal_loss(model, problem, grid, grid.to_grid(u))

    loss = cfg["w_f"] * l_pde + cfg["w_i"] * l_I
    return loss, {"pde": l_pde.item(), "I": l_I.item()}


# ===================================================================
# Training procedure
# ===================================================================

def train_pinn(config=None, verbose=True, reference=None):
    """Train a PINN on the transient nonlocal heat problem.

    Parameters
    ----------
    config : dict or None
        Overrides for DEFAULT_CONFIG.
    verbose : bool
    reference : dict or None
        Finite-difference reference (needed only when the problem has no
        exact solution).

    Returns
    -------
    dict with keys: 'model', 'history', 'errors', 'constraint_error',
                    'time', 'n_lbfgs_evals', 'lbfgs_failed', 'config'
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    problem = make_problem(cfg["problem"], cfg["alpha"], cfg["beta"], cfg["T"],
                           cfg["gamma"])
    if verbose:
        print(f"[PINN] Starting training on {problem}")

    model = HardConstrainedNet(problem, cfg["n_hidden"], cfg["n_layers"])
    model = model.to(DTYPE).to(DEVICE)
    grid = SpaceTimeGrid(problem, cfg["n_x"], cfg["n_t"])

    out = optimise(list(model.parameters()),
                   lambda: _loss_pinn(model, problem, grid, cfg),
                   cfg, verbose=verbose, label="PINN")

    errors = compute_errors(model, problem, reference=reference)

    if verbose:
        print(f"[PINN] Done in {out['time']:.1f}s "
              f"({out['n_lbfgs_evals']} L-BFGS evaluations)")
        print(f"  L2 = {errors['L2']:.2e}   Linf = {errors['Linf']:.2e}   "
              f"H1(T) = {errors['H1_T']:.2e}")
        print(f"  Nonlocal BC error (max over t) = {errors['nonlocal_max']:.2e}")
        print(f"  Flux -k u_x(0,t) by autograd, relative L2 error = "
              f"{errors['flux_ad_rel']:.2e}")

    return {
        "model": model,
        "history": out["history"],
        "errors": errors,
        "constraint_error": errors["nonlocal_max"],
        "time": out["time"],
        "n_lbfgs_evals": out["n_lbfgs_evals"],
        "lbfgs_failed": out["lbfgs_failed"],
        "config": cfg,
    }


# ===================================================================
# Quick test
# ===================================================================
if __name__ == "__main__":
    res = train_pinn({"n_adam": 1000, "n_lbfgs": 500}, verbose=True)
    print(f"\nFinal L2 error: {res['errors']['L2']:.2e}")
