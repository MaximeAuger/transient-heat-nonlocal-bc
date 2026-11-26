"""
Neural network shared by the PINN and the VPINN.

The same fully connected network (tanh, Xavier initialisation) as in the
steady-state benchmark is used, with the space-time input (x, t).  The
initial condition and the Dirichlet condition at x = 1 are imposed exactly,
and identically for both methods, through the ansatz

    u_theta(x, t) = u0(x) + [g1(t) - g1(0)] + (t / T) (1 - x) N_theta(x, t),

which gives u_theta(x, 0) = u0(x) and u_theta(1, t) = g1(t) (compatible data,
u0(1) = g1(0)).  The left boundary x = 0 is left free: it carries the
nonlocal integral condition, which is the object of the comparison.

Nonlocal initial condition (experiment E5).  For
u(x, 0) = gamma u(x, T) + phi(x), gamma != 1, write u = A(x) + G(t) + B(x, t)
with G(t) = g1(t) - g1(0) and B = (t / T)(1 - x) N_theta, both vanishing at
t = 0.  The condition A = gamma [A + G(T) + B(x, T)] + phi is solved for A:

    A(x) = [ phi(x) + gamma ( G(T) + B(x, T) ) ] / (1 - gamma),

so the nonlocal initial condition is also satisfied exactly, at the price of
one extra network evaluation at t = T.

References
----------
    - Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019). Physics-informed
      neural networks: A deep learning framework for solving forward and
      inverse problems involving nonlinear partial differential equations.
      J. Comput. Phys., 378, 686-707.

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron with tanh activations.

    Architecture
    ------------
        (x, t) -> [Linear -> Tanh] x n_layers -> Linear -> N_theta

    Parameters
    ----------
    n_input : int
        Input dimension (2 for a space-time point).
    n_output : int
        Output dimension (1 for a scalar field).
    n_hidden : int
        Number of neurons per hidden layer.
    n_layers : int
        Number of hidden layers.
    """

    def __init__(self, n_input: int = 2, n_output: int = 1,
                 n_hidden: int = 32, n_layers: int = 4):
        super().__init__()

        layers = [nn.Linear(n_input, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(n_hidden, n_output))

        self.net = nn.Sequential(*layers)
        self._initialise_weights()

    def _initialise_weights(self) -> None:
        """Xavier / Glorot normal initialisation for all linear layers."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class HardConstrainedNet(nn.Module):
    """Space-time approximation with exact initial and Dirichlet conditions.

    Parameters
    ----------
    problem : exact_solution.Problem
        Provides u0 (PyTorch), g1 (PyTorch) and the final time T.
    n_hidden, n_layers : int
        Size of the underlying MLP.
    """

    def __init__(self, problem, n_hidden: int = 32, n_layers: int = 4):
        super().__init__()
        self.problem = problem
        self.T = problem.T
        self.mlp = MLP(2, 1, n_hidden, n_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate u_theta at points x, t of shape (N, 1)."""
        if self.problem.gamma != 0.0:
            return self._forward_nonlocal_initial(x, t)
        z = torch.cat([2.0 * x - 1.0, 2.0 * t / self.T - 1.0], dim=1)
        lift = (self.problem.u0_torch(x)
                + self.problem.g1_torch(t) - self.problem.g1_at_0)
        return lift + (t / self.T) * (1.0 - x) * self.mlp(z)

    def _forward_nonlocal_initial(self, x, t):
        """Ansatz for u(x, 0) = gamma u(x, T) + phi(x), gamma != 1."""
        p, gamma = self.problem, self.problem.gamma
        if gamma == 1.0:
            raise ValueError("the hard-constraint ansatz requires gamma != 1")
        xi = 2.0 * x - 1.0
        b = (t / self.T) * (1.0 - x) * self.mlp(
            torch.cat([xi, 2.0 * t / self.T - 1.0], dim=1))
        b_T = (1.0 - x) * self.mlp(torch.cat([xi, torch.ones_like(x)], dim=1))
        g_T = p.g1_at_T - p.g1_at_0
        a = (p.ic_phi_torch(x) + gamma * (g_T + b_T)) / (1.0 - gamma)
        return a + p.g1_torch(t) - p.g1_at_0 + b
