"""
VPINN solver for the transient heat equation with a nonlocal integral
boundary condition (weak form in space, strong form in time).

PDE:  u_t - [k(u) u_x]_x = f(x, t),   k(u) = 1 + beta * u^2
BCs:  u(0, t) = alpha * int_0^1 u dx + g0(t),   u(1, t) = g1(t)
IC:   u(x, 0) = u0(x)

Weak formulation (Petrov-Galerkin).  Multiply by a product test function
v_i(x) w_j(t), integrate over Q = (0, 1) x (0, T) and integrate by parts in x
only (u_t is kept in strong form).  With v_i(1) = 0:

    int_Q [ u_t v_i + k(u) u_x v_i' ] w_j dx dt
        - v_i(0) int_0^T lambda(t) w_j(t) dt  =  int_Q f v_i w_j dx dt

where  lambda(t) = -k(u(0,t)) u_x(0,t)  is the heat flux entering through the
nonlocal boundary.  It is not known a priori and is treated as a Lagrange
multiplier, now a function of time:

    lambda(t) = sum_m c_m P_m(2t/T - 1),      c_m trainable.

Test functions: v_i(x) = P_i(2x-1) - 1  (Legendre, v_i(1) = 0),
                w_j(t) = P_j(2t/T - 1)  (Legendre).

Only first derivatives of the network are needed.  The initial and Dirichlet
conditions are imposed exactly by the network ansatz and the nonlocal
condition is the same penalty as in the PINN (utils.nonlocal_loss).

Loss:
    L = w_w * mean_ij(R_ij^2) + w_i * mean_t(r_I^2)

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import numpy as np
import torch

from exact_solution import make_problem, k_conductivity, DTYPE
from network import HardConstrainedNet
from utils import (
    COMMON_CONFIG, DEVICE, SpaceTimeGrid, LegendreTestFunctions,
    legendre_time_basis, nonlocal_loss, optimise, compute_errors,
)


# ===================================================================
# Default configuration
# ===================================================================

DEFAULT_CONFIG = {
    **COMMON_CONFIG,
    "n_test_x": 12,        # Legendre test functions in space (K_x)
    "n_test_t": 8,         # Legendre test functions in time  (K_t)
    "n_lambda": None,      # modes of lambda(t); None -> n_test_t
    "normalise_tests": "h1",   # "none", "h1" or "l2" (see utils.py)
    "w_w": 1000.0,         # weak residual weight
}


# ===================================================================
# Pre-computed data (quantities that do not change during training)
# ===================================================================

class VPINNData:
    """Pre-computed arrays for the VPINN loss evaluation.

    Attributes
    ----------
    Vw   : Tensor (n_x, K_x)   wx_a * v_i(x_a)
    dVw  : Tensor (n_x, K_x)   wx_a * v_i'(x_a)
    Ww   : Tensor (n_t, K_t)   wt_b * w_j(t_b)
    v0   : Tensor (K_x, 1)     v_i(0)
    P    : Tensor (n_t, M)     lambda basis P_m(2 t_b / T - 1)
    int_f : Tensor (K_x, K_t)  pre-computed int_Q f v_i w_j
    """

    def __init__(self, problem, grid, cfg):
        norm = cfg["normalise_tests"]
        tf = LegendreTestFunctions(cfg["n_test_x"], normalise=norm)
        norm = norm != "none"          # unit L2(0, T) norm in time
        self.n_lambda = cfg["n_lambda"] or cfg["n_test_t"]

        V = tf.eval_v(grid.xq)
        dV = tf.eval_dv(grid.xq)
        W = legendre_time_basis(cfg["n_test_t"], grid.tq, problem.T, norm)
        P = legendre_time_basis(self.n_lambda, grid.tq, problem.T)

        def tens(a):
            return torch.tensor(a, dtype=DTYPE, device=DEVICE)

        self.Vw = tens(grid.wx_np[:, None] * V)
        self.dVw = tens(grid.wx_np[:, None] * dV)
        self.Ww = tens(grid.wt_np[:, None] * W)
        self.v0 = tens(tf.v_at_zero().reshape(-1, 1))
        self.P = tens(P)
        self.int_f = self.Vw.T @ grid.f @ self.Ww


# ===================================================================
# Loss function
# ===================================================================

def _loss_vpinn(model, coeffs, problem, grid, vd, cfg):
    """Total VPINN loss (weak-form residual + nonlocal penalty).

    The nonlinear flux k(u) u_x depends on the current prediction and is
    re-evaluated at every step; int f v_i w_j is pre-computed.
    """
    beta = problem.beta
    x = grid.x.clone().requires_grad_(True)
    t = grid.t.clone().requires_grad_(True)

    u = model(x, t)
    u_x, u_t = torch.autograd.grad(
        u, [x, t], grad_outputs=torch.ones_like(u), create_graph=True)
    flux = grid.to_grid(k_conductivity(u, beta) * u_x)   # k(u) u_x, (n_x, n_t)
    u_t = grid.to_grid(u_t)

    # R_ij = int (u_t v_i + k u_x v_i') w_j - v_i(0) int lambda w_j - int f v_i w_j
    lam = vd.P @ coeffs                                   # lambda(t_b), (n_t,)
    R = (vd.Vw.T @ u_t @ vd.Ww + vd.dVw.T @ flux @ vd.Ww
         - vd.v0 * (lam @ vd.Ww).unsqueeze(0) - vd.int_f)  # (K_x, K_t)
    l_w = torch.mean(R ** 2)

    # --- Nonlocal condition: u(0,t) = alpha * int u dx + g0(t) ---
    l_I = nonlocal_loss(model, problem, grid, grid.to_grid(u))

    loss = cfg["w_w"] * l_w + cfg["w_i"] * l_I
    return loss, {"weak": l_w.item(), "I": l_I.item()}


# ===================================================================
# Training procedure
# ===================================================================

def train_vpinn(config=None, verbose=True, reference=None):
    """Train a VPINN on the transient nonlocal heat problem.

    Parameters
    ----------
    config : dict or None
    verbose : bool
    reference : dict or None
        Finite-difference reference (needed only when the problem has no
        exact solution).

    Returns
    -------
    dict with keys: 'model', 'lam_coeffs', 'lam_fn', 'history', 'errors',
                    'constraint_error', 'time', 'n_lbfgs_evals',
                    'lbfgs_failed', 'config'
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    problem = make_problem(cfg["problem"], cfg["alpha"], cfg["beta"], cfg["T"])
    if verbose:
        print(f"[VPINN] Starting training on {problem}")

    # --- Network + Lagrange multiplier lambda(t) ---
    model = HardConstrainedNet(problem, cfg["n_hidden"], cfg["n_layers"])
    model = model.to(DTYPE).to(DEVICE)
    grid = SpaceTimeGrid(problem, cfg["n_x"], cfg["n_t"])
    vd = VPINNData(problem, grid, cfg)
    coeffs = torch.nn.Parameter(
        torch.zeros(vd.n_lambda, dtype=DTYPE, device=DEVICE))

    out = optimise(list(model.parameters()) + [coeffs],
                   lambda: _loss_vpinn(model, coeffs, problem, grid, vd, cfg),
                   cfg, verbose=verbose, label="VPINN")

    c = coeffs.detach().cpu().numpy()

    def lam_fn(t):
        """Learned boundary flux lambda(t)."""
        return legendre_time_basis(len(c), t, problem.T) @ c

    errors = compute_errors(model, problem, lam_fn=lam_fn, reference=reference)

    if verbose:
        print(f"[VPINN] Done in {out['time']:.1f}s "
              f"({out['n_lbfgs_evals']} L-BFGS evaluations)")
        print(f"  L2 = {errors['L2']:.2e}   Linf = {errors['Linf']:.2e}   "
              f"H1(T) = {errors['H1_T']:.2e}")
        print(f"  Nonlocal BC error (max over t) = {errors['nonlocal_max']:.2e}")
        print(f"  Learned lambda(t), relative L2 error = "
              f"{errors['flux_lam_rel']:.2e}   "
              f"(autograd flux: {errors['flux_ad_rel']:.2e})")

    return {
        "model": model,
        "lam_coeffs": c,
        "lam_fn": lam_fn,
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
    res = train_vpinn({"n_adam": 1000, "n_lbfgs": 500}, verbose=True)
    print(f"\nFinal L2 error: {res['errors']['L2']:.2e}")
