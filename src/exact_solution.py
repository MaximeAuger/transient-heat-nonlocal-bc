"""
Problem definitions for the transient heat equation with a nonlocal integral
boundary condition (Cannon-type problem).

PDE:  u_t - [k(u) u_x]_x = f(x, t),     x in (0, 1),  t in (0, T]
      k(u) = 1 + beta * u^2             (beta = 0: linear case)

BCs:  u(0, t) = alpha * int_0^1 u(x, t) dx + g0(t)    (nonlocal integral BC)
      u(1, t) = g1(t)                                 (Dirichlet)
IC:   u(x, 0) = u0(x)

All data (f, g0, g1, u0) are derived symbolically with SymPy from a
manufactured solution, for any (alpha, beta), and then compiled to NumPy and
PyTorch callables.  Nothing is hard-coded by hand.

Manufactured solutions
----------------------
    "decay"       u = exp(-t) * phi(x)
    "oscillating" u = (1 + t + 0.5 sin(2 pi t)) * phi(x)     (non-monotone in t)

with the spatial profile of the steady-state benchmark

    phi(x) = sin(pi x) + (1 - x) * 4 / pi,

for which phi(0) = int_0^1 phi dx = 4/pi, hence g0 = 0 when alpha = 1.

A source-free problem without closed-form solution (f = 0, g0 = g1 = 0) is
also provided; it is used with the finite-difference reference.

Optional extension (experiment E5): the local initial condition can be
replaced by a nonlocal one of Sapagovas type,

    u(x, 0) = gamma * u(x, T) + phi(x),

with phi derived from the manufactured solution (gamma = 0: usual problem).

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import math

import numpy as np
import sympy as sp
import torch

DTYPE = torch.float64

# Symbols shared by every problem
X, TT = sp.symbols("x t", real=True)

# Default physical parameters
ALPHA_DEFAULT = 1.0
BETA_DEFAULT = 0.0
T_DEFAULT = 1.0

# Spatial profile of the steady-state benchmark
PHI = sp.sin(sp.pi * X) + (1 - X) * 4 / sp.pi

_TORCH_NAMESPACE = {
    "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
    "exp": torch.exp, "sinh": torch.sinh, "cosh": torch.cosh,
    "tanh": torch.tanh, "sqrt": torch.sqrt, "log": torch.log,
    "pi": math.pi,
}


# ---------------------------------------------------------------------------
# Conductivity  k(u) = 1 + beta * u^2
# ---------------------------------------------------------------------------

def k_conductivity(u, beta=BETA_DEFAULT):
    """Nonlinear conductivity coefficient (works for NumPy and PyTorch)."""
    return 1.0 + beta * u ** 2


def dk_conductivity(u, beta=BETA_DEFAULT):
    """Derivative dk/du = 2 * beta * u."""
    return 2.0 * beta * u


# ---------------------------------------------------------------------------
# SymPy -> NumPy / PyTorch compilation helpers
# ---------------------------------------------------------------------------

def _exact_rational(value):
    """Convert a float such as 1.99 to the exact rational 199/100."""
    return sp.Rational(str(value))


def _numpy_fn(args, expr):
    """Vectorised NumPy callable; constants are broadcast to the input shape."""
    fn = sp.lambdify(args, expr, modules="numpy")

    def wrapped(*vals):
        vals = [np.asarray(v, dtype=float) for v in vals]
        out = np.asarray(fn(*vals), dtype=float)
        shape = np.broadcast(*vals).shape
        return np.broadcast_to(out, shape).copy()

    return wrapped


def _torch_fn(args, expr):
    """PyTorch callable (differentiable); constants are broadcast."""
    # symbol-free calls such as exp(-1) are evaluated once, as floats
    constants = [a for a in sp.sympify(expr).atoms(sp.Function)
                 if not a.free_symbols]
    expr = sp.sympify(expr).xreplace({c: sp.Float(c.evalf(20)) for c in constants})
    fn = sp.lambdify(args, expr, modules=[_TORCH_NAMESPACE])

    def wrapped(*vals):
        out = fn(*vals)
        if not torch.is_tensor(out):
            out = torch.full_like(vals[0], float(out))
        return out

    return wrapped


# ---------------------------------------------------------------------------
# Problem container
# ---------------------------------------------------------------------------

class Problem:
    """Initial-boundary value problem with a nonlocal integral condition.

    Parameters
    ----------
    name : str
    alpha, beta, T : float
        Nonlocal coefficient, nonlinearity strength and final time.
    f, g0, g1, u0 : sympy expressions in (x, t), (t), (t), (x)
        Problem data.
    u : sympy expression or None
        Exact solution when available (manufactured problems).

    NumPy API   : f(x,t), g0(t), g1(t), u0(x) and, if an exact solution is
                  known, u(x,t), u_x(x,t), u_t(x,t), integral(t), flux0(t).
    PyTorch API : u0_torch(x), g1_torch(t)  (used by the hard-constraint
                  ansatz) and u_torch(x,t) when available.
    """

    def __init__(self, name, alpha, beta, T, f, g0, g1, u0, u=None,
                 gamma=0.0, ic_phi=None):
        self.name = name
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.T = float(T)
        self.gamma = float(gamma)
        self.sym = {"f": f, "g0": g0, "g1": g1, "u0": u0, "u": u,
                    "ic_phi": ic_phi}
        if self.gamma != 0.0:
            # nonlocal initial condition u(x,0) = gamma u(x,T) + phi(x)
            self.ic_phi = _numpy_fn((X,), ic_phi)
            self.ic_phi_torch = _torch_fn((X,), ic_phi)
            self.g1_at_T = float(sp.N(g1.subs(TT, _exact_rational(T))))

        self.f = _numpy_fn((X, TT), f)
        self.g0 = _numpy_fn((TT,), g0)
        self.g1 = _numpy_fn((TT,), g1)
        self.u0 = _numpy_fn((X,), u0)
        self.u0_x = _numpy_fn((X,), sp.diff(u0, X))

        self.u0_torch = _torch_fn((X,), u0)
        self.g1_torch = _torch_fn((TT,), g1)
        self.g1_at_0 = float(sp.N(g1.subs(TT, 0)))

        self.has_exact = u is not None
        if self.has_exact:
            u_x = sp.diff(u, X)
            k = 1 + _exact_rational(beta) * u ** 2
            self.u = _numpy_fn((X, TT), u)
            self.u_x = _numpy_fn((X, TT), u_x)
            self.u_t = _numpy_fn((X, TT), sp.diff(u, TT))
            self.integral = _numpy_fn((TT,), sp.integrate(u, (X, 0, 1)))
            # Flux entering the domain at the left boundary: -k(u) u_x (0, t)
            self.flux0 = _numpy_fn((TT,), (-k * u_x).subs(X, 0))
            self.u_torch = _torch_fn((X, TT), u)

    def __repr__(self):
        extra = f", gamma={self.gamma:g}" if self.gamma != 0.0 else ""
        return (f"Problem({self.name}, alpha={self.alpha:g}, "
                f"beta={self.beta:g}, T={self.T:g}{extra})")


# ---------------------------------------------------------------------------
# Manufactured problems
# ---------------------------------------------------------------------------

def manufactured_expression(kind="decay"):
    """Return the symbolic manufactured solution u(x, t)."""
    if kind == "decay":
        return sp.exp(-TT) * PHI
    if kind == "oscillating":
        return (1 + TT + sp.Rational(1, 2) * sp.sin(2 * sp.pi * TT)) * PHI
    raise ValueError(f"unknown manufactured solution '{kind}'")


def manufactured(kind="decay", alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT,
                 T=T_DEFAULT, gamma=0.0):
    """Build a manufactured problem: data are derived from u by substitution.

        f   = u_t - [k(u) u_x]_x
        g0  = u(0, t) - alpha * int_0^1 u dx
        g1  = u(1, t)
        u0  = u(x, 0)
        phi = u(x, 0) - gamma * u(x, T)      (nonlocal initial condition)
    """
    u = manufactured_expression(kind)
    a, b = _exact_rational(alpha), _exact_rational(beta)
    k = 1 + b * u ** 2
    f = sp.diff(u, TT) - sp.diff(k * sp.diff(u, X), X)
    g0 = sp.simplify(u.subs(X, 0) - a * sp.integrate(u, (X, 0, 1)))
    g1 = sp.simplify(u.subs(X, 1))
    u0 = u.subs(TT, 0)
    ic_phi = u0 - _exact_rational(gamma) * u.subs(TT, _exact_rational(T))
    return Problem(kind, alpha, beta, T, f, g0, g1, u0, u=u,
                   gamma=gamma, ic_phi=ic_phi)


def source_free(alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT, T=T_DEFAULT):
    """Source-free problem without closed-form solution (f = 0, g0 = g1 = 0).

    The initial state u0 = sin(pi x) + c (1 - x) is made compatible with the
    nonlocal condition, u0(0) = alpha * int u0, which requires

        c = 2 alpha / ( pi (1 - alpha / 2) ),

    singular at alpha = 2 (the critical value, see critical_alpha.py).
    """
    a = _exact_rational(alpha)
    if a == 2:
        raise ValueError("no compatible initial state of this form at alpha = 2")
    c = 2 * a / (sp.pi * (1 - a / 2))
    u0 = sp.sin(sp.pi * X) + c * (1 - X)
    zero = sp.Integer(0)
    return Problem("source_free", alpha, beta, T, zero, zero, zero, u0, u=None)


def make_problem(kind="decay", alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT,
                 T=T_DEFAULT, gamma=0.0):
    """Factory used by the solvers and by run_all.py."""
    if kind == "source_free":
        if gamma != 0.0:
            raise ValueError("source_free is defined for gamma = 0 only")
        return source_free(alpha, beta, T)
    return manufactured(kind, alpha, beta, T, gamma)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(problem, n=41, verbose=True):
    """Check a manufactured problem numerically (independently of SymPy).

      1. PDE residual  u_t - [k(u) u_x]_x - f  via PyTorch autograd,
      2. nonlocal condition with a Gauss-Legendre rule,
      3. Dirichlet and initial conditions.
    """
    assert problem.has_exact
    beta, T = problem.beta, problem.T
    xs = torch.linspace(0.0, 1.0, n, dtype=DTYPE)
    ts = torch.linspace(0.0, T, n, dtype=DTYPE)
    xg, tg = torch.meshgrid(xs, ts, indexing="ij")
    x = xg.reshape(-1, 1).clone().requires_grad_(True)
    t = tg.reshape(-1, 1).clone().requires_grad_(True)

    u = problem.u_torch(x, t)
    ones = torch.ones_like(u)
    u_x, u_t = torch.autograd.grad(u, [x, t], ones, create_graph=True)
    flux = k_conductivity(u, beta) * u_x
    flux_x = torch.autograd.grad(flux, x, ones)[0]
    f = problem.f(x.detach().numpy(), t.detach().numpy())
    res = (u_t - flux_x).detach().numpy() - f
    scale = max(1.0, np.max(np.abs(f)))
    pde_err = np.max(np.abs(res)) / scale

    xi, wi = np.polynomial.legendre.leggauss(40)
    xq, wq = 0.5 * (xi + 1.0), 0.5 * wi
    tn = np.linspace(0.0, T, n)
    integ = np.array([np.sum(wq * problem.u(xq, np.full_like(xq, ti)))
                      for ti in tn])
    u_left = problem.u(np.zeros_like(tn), tn)
    nl_err = np.max(np.abs(u_left - problem.alpha * integ - problem.g0(tn)))
    int_err = np.max(np.abs(integ - problem.integral(tn)))
    d_err = np.max(np.abs(problem.u(np.ones_like(tn), tn) - problem.g1(tn)))
    xn = np.linspace(0.0, 1.0, n)
    ic_err = np.max(np.abs(problem.u(xn, np.zeros_like(xn)) - problem.u0(xn)))

    out = {"pde": pde_err, "nonlocal": nl_err, "integral": int_err,
           "dirichlet": d_err, "initial": ic_err}
    ok = all(v < 1e-10 for v in out.values())
    if verbose:
        print(f"=== Verification: {problem} ===")
        print(f"  PDE residual (autograd, relative) : {pde_err:.2e}")
        print(f"  Nonlocal condition error          : {nl_err:.2e}")
        print(f"  Integral (quadrature vs SymPy)    : {int_err:.2e}")
        print(f"  Dirichlet error                   : {d_err:.2e}")
        print(f"  Initial condition error           : {ic_err:.2e}")
        print(f"  g0(t) = {problem.sym['g0']},   g1(t) = {problem.sym['g1']}")
        print(f"  Status: {'PASS' if ok else 'FAIL'}")
    out["ok"] = ok
    return out


if __name__ == "__main__":
    for kind in ("decay", "oscillating"):
        for alpha, beta in [(1.0, 0.0), (1.0, 1.0), (1.99, 0.0), (3.0, 5.0)]:
            verify(manufactured(kind, alpha, beta))
            print()
