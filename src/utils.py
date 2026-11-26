"""
Utility functions shared by the PINN and VPINN solvers.

Contents
--------
- Gauss-Legendre quadrature
- Legendre test functions (space) and Legendre basis (time)
- Space-time quadrature grid, common to both methods
- Nonlocal integral condition residual, common to both methods
- Automatic differentiation helper
- Adam + L-BFGS training loop, common to both methods
- Error metrics (space-time L2 / Linf, H1 at t = T, nonlocal condition, flux)

Everything that enters the comparison (points, quadratures, nonlocal penalty,
optimiser, metrics) lives here so that the two solvers differ only in the way
the PDE residual is measured.

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import os
import time

import numpy as np
import torch
import torch.optim as optim
from numpy.polynomial import legendre as leg

from exact_solution import DTYPE, k_conductivity

DEVICE = torch.device(
    os.environ.get("THNBC_DEVICE")
    or ("cuda" if torch.cuda.is_available() else "cpu"))

# Settings shared by both solvers (identical network, points and budget)
COMMON_CONFIG = {
    "problem": "decay",    # "decay", "oscillating" or "source_free"
    "alpha": 1.0,          # nonlocal coefficient
    "beta": 0.0,           # nonlinearity strength, k(u) = 1 + beta u^2
    "T": 1.0,              # final time
    "gamma": 0.0,          # nonlocal initial condition u(x,0) = gamma u(x,T) + phi
    "n_hidden": 32,
    "n_layers": 4,
    "n_x": 32,             # Gauss-Legendre points in space
    "n_t": 32,             # Gauss-Legendre points in time
    "w_i": 100.0,          # nonlocal integral condition weight
    "lr_adam": 1e-3,
    "n_adam": 4_000,       # Adam iterations
    "n_lbfgs": 4_000,      # L-BFGS budget (closure evaluations)
    "seed": 0,
}


# ===================================================================
# 1. Gauss-Legendre quadrature on [a, b]
# ===================================================================

def gauss_legendre(n, a=0.0, b=1.0):
    """Return nodes and weights of an *n*-point Gauss-Legendre rule on [a, b]."""
    xi, wi = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * xi + 0.5 * (a + b)
    w = 0.5 * (b - a) * wi
    return x, w


# ===================================================================
# 2. Legendre test functions
# ===================================================================

class LegendreTestFunctions:
    """Shifted Legendre test functions in space, v_k(x) = P_k(2x-1) - 1.

    These satisfy v_k(1) = 0 (the Dirichlet boundary) and are free at x = 0,
    where the boundary flux appears as a Lagrange multiplier.

    Parameters
    ----------
    n_test : int
        Number of test functions (k = 1, ..., n_test).
    normalise : {"none", "h1", "l2"}
        "none" : raw functions, as in the steady-state code;
        "h1"   : v_k / ||v_k'||_{L2(0,1)},  ||v_k'||^2 = 2 k (k + 1);
        "l2"   : v_k / ||P_k||_{L2(0,1)},   ||P_k||^2  = 1 / (2 k + 1).
        The scaling only changes the relative weight of the weak residuals.
    """

    def __init__(self, n_test: int, normalise: str = "none"):
        self.n_test = n_test
        k = np.arange(1, n_test + 1)
        if normalise == "h1":
            self.scale = 1.0 / np.sqrt(2.0 * k * (k + 1.0))
        elif normalise == "l2":
            self.scale = np.sqrt(2.0 * k + 1.0)
        elif normalise == "none":
            self.scale = np.ones(n_test)
        else:
            raise ValueError(f"unknown normalisation '{normalise}'")

    def eval_v(self, x01):
        """Evaluate all test functions at points in [0, 1]."""
        xi = 2.0 * np.asarray(x01) - 1.0
        V = np.zeros((len(xi), self.n_test))
        for k in range(1, self.n_test + 1):
            c = np.zeros(k + 1)
            c[k] = 1.0
            V[:, k - 1] = leg.legval(xi, c) - 1.0
        return V * self.scale

    def eval_dv(self, x01):
        """Evaluate derivatives: dv_k/dx = 2 * P_k'(2x - 1)."""
        xi = 2.0 * np.asarray(x01) - 1.0
        dV = np.zeros((len(xi), self.n_test))
        for k in range(1, self.n_test + 1):
            c = np.zeros(k + 1)
            c[k] = 1.0
            dV[:, k - 1] = 2.0 * leg.legval(xi, leg.legder(c))
        return dV * self.scale

    def v_at_zero(self):
        """Return v_k(0) = P_k(-1) - 1 = (-1)^k - 1."""
        k = np.arange(1, self.n_test + 1)
        return ((-1.0) ** k - 1.0) * self.scale


def legendre_time_basis(n, t, T=1.0, normalise=False):
    """Legendre polynomials P_0..P_{n-1}(2t/T - 1) at times t, shape (N, n).

    Used both as time test functions w_j(t) and as the expansion basis of the
    boundary flux lambda(t).  With normalise=True each column has unit
    L2(0, T) norm (||P_j||^2 = T / (2j + 1)).
    """
    s = 2.0 * np.asarray(t) / T - 1.0
    B = leg.legvander(s, n - 1)
    if normalise:
        B = B * np.sqrt((2.0 * np.arange(n) + 1.0) / T)
    return B


# ===================================================================
# 3. Space-time quadrature grid (common to PINN and VPINN)
# ===================================================================

class SpaceTimeGrid:
    """Tensor Gauss-Legendre grid on (0, 1) x (0, T).

    Both solvers evaluate the network at exactly these n_x * n_t points; the
    spatial rule is also the one used for the nonlocal integral.

    Attributes
    ----------
    x, t  : Tensor (n_x * n_t, 1)   grid points (x-major ordering)
    wx    : Tensor (n_x, 1)         spatial weights
    wt    : Tensor (n_t,)           temporal weights
    f     : Tensor (n_x, n_t)       source term on the grid
    t_nodes, x_left : Tensor (n_t, 1)  time nodes and the points (0, t_b)
    g0    : Tensor (n_t,)           nonlocal data g0(t_b)
    """

    def __init__(self, problem, n_x, n_t):
        self.n_x, self.n_t, self.T = n_x, n_t, problem.T
        xq, wx = gauss_legendre(n_x, 0.0, 1.0)
        tq, wt = gauss_legendre(n_t, 0.0, problem.T)
        self.xq, self.tq, self.wx_np, self.wt_np = xq, tq, wx, wt
        xg, tg = np.meshgrid(xq, tq, indexing="ij")

        def tens(a):
            return torch.tensor(a, dtype=DTYPE, device=DEVICE)

        self.x = tens(xg.reshape(-1, 1))
        self.t = tens(tg.reshape(-1, 1))
        self.wx = tens(wx.reshape(-1, 1))
        self.wt = tens(wt)
        self.f = tens(problem.f(xg, tg))
        self.t_nodes = tens(tq.reshape(-1, 1))
        self.x_left = torch.zeros_like(self.t_nodes)
        self.g0 = tens(problem.g0(tq))

    def to_grid(self, values):
        """Reshape a (n_x * n_t, 1) tensor to (n_x, n_t)."""
        return values.reshape(self.n_x, self.n_t)


def nonlocal_residual(model, problem, grid, u_grid):
    """Residual of the nonlocal condition at the time nodes, shape (n_t,).

        r_I(t_b) = u(0, t_b) - alpha * sum_a wx_a u(x_a, t_b) - g0(t_b)

    The integral reuses the network values already computed on the grid.
    """
    integral = torch.sum(grid.wx * u_grid, dim=0)
    u_left = model(grid.x_left, grid.t_nodes).squeeze(1)
    return u_left - problem.alpha * integral - grid.g0


def nonlocal_loss(model, problem, grid, u_grid):
    """Time-averaged squared nonlocal residual (Gauss-Legendre in time)."""
    r = nonlocal_residual(model, problem, grid, u_grid)
    return torch.sum(grid.wt * r ** 2) / grid.T


# ===================================================================
# 4. Automatic differentiation helper
# ===================================================================

def diff(u, x):
    """du/dx via automatic differentiation (graph kept for higher orders)."""
    return torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True)[0]


# ===================================================================
# 5. Training loop: Adam, then L-BFGS (same budget for both methods)
# ===================================================================

def optimise(params, loss_fn, cfg, verbose=True, label=""):
    """Minimise loss_fn with Adam followed by L-BFGS.

    Parameters
    ----------
    params : list of Tensors
    loss_fn : callable () -> (loss Tensor, dict of float components)
    cfg : dict with 'lr_adam', 'n_adam', 'n_lbfgs'
        n_lbfgs is a budget in closure (loss + gradient) evaluations.

    Returns
    -------
    dict with 'history' (lists), 'time', 'n_lbfgs_evals', 'lbfgs_failed'.
    """
    hist = {"loss": []}

    def record(loss, parts):
        hist["loss"].append(loss.item())
        for name, val in parts.items():
            hist.setdefault(name, []).append(val)

    t0 = time.time()

    # ---------------- Phase 1: Adam ----------------
    optimizer = optim.Adam(params, lr=cfg["lr_adam"])
    for epoch in range(cfg["n_adam"]):
        optimizer.zero_grad()
        loss, parts = loss_fn()
        loss.backward()
        optimizer.step()
        record(loss, parts)
        if verbose and (epoch + 1) % 1000 == 0:
            msg = "  ".join(f"{k}={v:.3e}" for k, v in parts.items())
            print(f"  [{label}] Adam  {epoch + 1:6d}/{cfg['n_adam']}  "
                  f"loss={loss.item():.3e}  {msg}")

    # ---------------- Phase 2: L-BFGS ----------------
    backup = [p.detach().clone() for p in params]
    optimizer2 = optim.LBFGS(
        params, lr=1.0, max_iter=20, history_size=50,
        tolerance_grad=1e-15, tolerance_change=1e-15,
        line_search_fn="strong_wolfe")
    n_eval = [0]
    last = [None]

    def closure():
        optimizer2.zero_grad()
        loss, parts = loss_fn()
        loss.backward()
        record(loss, parts)
        n_eval[0] += 1
        last[0] = (loss.item(), parts)
        return loss

    failed = False
    stall, previous = 0, None
    while n_eval[0] < cfg["n_lbfgs"]:
        before = n_eval[0]
        optimizer2.step(closure)
        current = last[0][0]
        if not np.isfinite(current):
            failed = True
            break
        if verbose and (n_eval[0] // 500) > (before // 500):
            msg = "  ".join(f"{k}={v:.3e}" for k, v in last[0][1].items())
            print(f"  [{label}] LBFGS {n_eval[0]:6d}/{cfg['n_lbfgs']}  "
                  f"loss={current:.3e}  {msg}")
        if previous is not None and abs(previous - current) <= 1e-12 * abs(previous):
            stall += 1
            if stall >= 3:
                break
        else:
            stall = 0
        previous = current

    if failed:  # restore the state reached by Adam and report the failure
        with torch.no_grad():
            for p, b in zip(params, backup):
                p.copy_(b)

    return {"history": hist, "time": time.time() - t0,
            "n_lbfgs_evals": n_eval[0], "lbfgs_failed": failed}


# ===================================================================
# 6. Error metrics
# ===================================================================

def _trapz2(values, x, t):
    return np.trapezoid(np.trapezoid(values, t, axis=1), x)


def compute_errors(model, problem, lam_fn=None, reference=None, n_eval=201):
    """Compute all error metrics on a uniform n_eval x n_eval space-time grid.

    The reference is the exact manufactured solution when available,
    otherwise a finite-difference solution `reference` (output of
    fd_reference.solve_crank_nicolson) whose grid contains the evaluation
    grid.

    Returns
    -------
    dict with arrays 'x', 't', 'u_ref', 'u_pred', 'error', 'flux_ref',
    'flux_ad' (and 'flux_lam' if lam_fn is given) and scalars
        'L2', 'L2_rel', 'Linf'      space-time errors,
        'H1_T', 'H1_semi_T'         H1 error at the final time,
        'nonlocal_max', 'nonlocal_L2'   violation of the integral condition,
        'flux_ad_L2', 'flux_ad_rel'     error of -k(u) u_x(0, t) by autograd,
        'flux_lam_L2', 'flux_lam_rel'   error of the learned lambda(t).
    """
    T, beta = problem.T, problem.beta
    x_np = np.linspace(0.0, 1.0, n_eval)
    t_np = np.linspace(0.0, T, n_eval)
    xg, tg = np.meshgrid(x_np, t_np, indexing="ij")

    def tens(a, grad=False):
        return torch.tensor(np.asarray(a).reshape(-1, 1), dtype=DTYPE,
                            device=DEVICE, requires_grad=grad)

    # --- reference fields ---
    if problem.has_exact:
        u_ref = problem.u(xg, tg)
        ux_ref_T = problem.u_x(x_np, np.full_like(x_np, T))
        flux_ref = problem.flux0(t_np)
    else:
        sx = (len(reference["x"]) - 1) // (n_eval - 1)
        st = (len(reference["t"]) - 1) // (n_eval - 1)
        assert np.allclose(reference["x"][::sx], x_np)
        assert np.allclose(reference["t"][::st], t_np)
        u_ref = reference["u"][::sx, ::st]
        ux_ref_T = np.gradient(reference["u"][:, -1], reference["x"],
                               edge_order=2)[::sx]
        flux_ref = reference["flux0"][::st]

    # --- space-time errors ---
    with torch.no_grad():
        u_pred = model(tens(xg), tens(tg)).cpu().numpy().reshape(xg.shape)
    err = u_pred - u_ref
    L2 = np.sqrt(_trapz2(err ** 2, x_np, t_np))
    out = {
        "x": x_np, "t": t_np, "u_ref": u_ref, "u_pred": u_pred, "error": err,
        "L2": L2,
        "L2_rel": L2 / np.sqrt(_trapz2(u_ref ** 2, x_np, t_np)),
        "Linf": np.max(np.abs(err)),
    }

    # --- H1 error at the final time ---
    x_T = tens(x_np, grad=True)
    u_T = model(x_T, torch.full_like(x_T, T))
    ux_T = diff(u_T, x_T).detach().cpu().numpy().ravel()
    e_T = u_T.detach().cpu().numpy().ravel() - u_ref[:, -1]
    semi = np.sqrt(np.trapezoid((ux_T - ux_ref_T) ** 2, x_np))
    out["H1_semi_T"] = semi
    out["H1_T"] = np.sqrt(np.trapezoid(e_T ** 2, x_np) + semi ** 2)

    # --- nonlocal condition: u(0,t) - alpha int u dx - g0(t) ---
    xq, wq = gauss_legendre(64)
    xqg, tqg = np.meshgrid(xq, t_np, indexing="ij")
    with torch.no_grad():
        u_q = model(tens(xqg), tens(tqg)).cpu().numpy().reshape(xqg.shape)
    viol = u_pred[0] - problem.alpha * (wq @ u_q) - problem.g0(t_np)
    out["nonlocal_max"] = np.max(np.abs(viol))
    out["nonlocal_L2"] = np.sqrt(np.trapezoid(viol ** 2, t_np))

    # --- boundary flux  -k(u) u_x (0, t) ---
    x0 = tens(np.zeros_like(t_np), grad=True)
    u0 = model(x0, tens(t_np))
    flux_ad = (-k_conductivity(u0, beta) * diff(u0, x0)).detach().cpu().numpy().ravel()
    ref_norm = np.sqrt(np.trapezoid(flux_ref ** 2, t_np))
    out["flux_ref"], out["flux_ad"] = flux_ref, flux_ad
    out["flux_ad_L2"] = np.sqrt(np.trapezoid((flux_ad - flux_ref) ** 2, t_np))
    out["flux_ad_rel"] = out["flux_ad_L2"] / ref_norm
    if lam_fn is not None:
        flux_lam = lam_fn(t_np)
        out["flux_lam"] = flux_lam
        out["flux_lam_L2"] = np.sqrt(np.trapezoid((flux_lam - flux_ref) ** 2, t_np))
        out["flux_lam_rel"] = out["flux_lam_L2"] / ref_norm
    return out
