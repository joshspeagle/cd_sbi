"""DoublyMonotoneUMNN — §6.1 form 2:

    r(θ, T) = b_umnn(T) + ∫_{θ_ref}^{θ} softplus(α(t) + β_umnn(T)) dt

Both R1 and R2 enforced architecturally via Gauss-Legendre quadrature.
The §8.4 main flow; the closed-form log|∂_T r| accompanies the forward
pass so NF-MLE's Jacobian factor never needs autograd through the flow's
internals.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cdsbi.flows.base import Flow, Guarantee


_NODES_NP, _WEIGHTS_NP = np.polynomial.legendre.leggauss(12)


class _MonotoneScalarUMNN(nn.Module):
    """A 1D monotone-increasing function of T via T → bias + ∫_0^T softplus(MLP(t)) dt.

    Self-contained quadrature; does not depend on v0's UMNNBlock to keep the
    integral signature clean (we use both the value and the derivative).
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _integrand(self, t: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.mlp(t)) + 1e-3  # strictly positive

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        """∫_0^T softplus(MLP(t)) dt + bias, with T positive."""
        n = T.shape[0]
        # Map [-1, 1] nodes to [0, T]
        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)
        T_exp = T.view(n, 1, 1).expand(-1, u.size(1), -1)
        t = 0.5 * T_exp * (u + 1.0)
        integrand = self._integrand(t)
        weights = self._weights.view(1, -1, 1)
        integral = 0.5 * T * (weights * integrand).sum(dim=1)
        return self.bias + integral

    def derivative(self, T: torch.Tensor) -> torch.Tensor:
        """f'(T) = softplus(MLP(T))."""
        return self._integrand(T)


class DoublyMonotoneUMNN(nn.Module, Flow):
    monotonicity_guarantees = frozenset({Guarantee.R1, Guarantee.R2})

    def __init__(self, hidden: int = 16, theta_ref: float = 0.3):
        super().__init__()
        self.hidden = hidden
        self.theta_ref = theta_ref
        # b_umnn(T): scalar monotone-increasing function of T
        self._b_umnn = _MonotoneScalarUMNN(hidden=hidden)
        # β_umnn(T): scalar monotone-increasing function of T (parameter inside the integrand)
        self._beta_umnn = _MonotoneScalarUMNN(hidden=hidden)
        # α(t): scalar trainable bias on the θ-integrand
        # Implemented as a small MLP for flexibility; for the §8.4 truth pivot the
        # learned α should saturate at the value that makes ∂_θ r match
        # ∂_θ Φ⁻¹(F_{χ²_{2n}}(2θT)) on average.
        self._alpha_net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )
        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _integrand(self, t: torch.Tensor, beta_T: torch.Tensor) -> torch.Tensor:
        """softplus(α(t) + β_umnn(T)) — strictly positive integrand."""
        alpha_t = self._alpha_net(t)
        return F.softplus(alpha_t + beta_T) + 1e-3

    def forward(
        self, theta: torch.Tensor, context: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """r(θ, T) and log|∂r/∂T| in closed form. context = T, shape (n, 1)."""
        assert context is not None and context.shape[-1] == 1, (
            "DoublyMonotoneUMNN expects context = T scalar, shape (n, 1)"
        )
        T = context
        n = theta.shape[0]

        # β_umnn(T) and its derivative; constant in the θ-integrand
        beta_T = self._beta_umnn(T)             # (n, 1)
        beta_prime_T = self._beta_umnn.derivative(T)  # (n, 1)

        # b_umnn(T) and b'_umnn(T) — used in the constant term and the Jacobian
        b_T = self._b_umnn(T)                    # (n, 1)
        b_prime_T = self._b_umnn.derivative(T)   # (n, 1)

        # Map [-1, 1] nodes to [theta_ref, theta]
        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)  # (n, K, 1)
        a = self.theta_ref
        b = theta.view(n, 1, 1).expand(-1, u.size(1), -1)
        # t = a + (b - a)/2 * (u + 1)
        t = a + 0.5 * (b - a) * (u + 1.0)

        # β_T broadcast across nodes
        beta_exp = beta_T.unsqueeze(1).expand(-1, u.size(1), -1)  # (n, K, 1)
        integrand = self._integrand(t.reshape(-1, 1), beta_exp.reshape(-1, 1)).view(n, -1, 1)

        weights = self._weights.view(1, -1, 1)
        # ∫_{theta_ref}^{theta} softplus(α(t) + β_T) dt = (theta - theta_ref) / 2 · Σ w_i f(t_i)
        integral = 0.5 * (theta - a) * (weights * integrand).sum(dim=1)  # (n, 1)

        r = b_T + integral  # (n, 1)

        # ∂_T r = b'_umnn(T) + β'_umnn(T) · ∫ σ(α(t) + β_T) dt
        # where σ = derivative of softplus = sigmoid. By the same quadrature:
        sigmoid_integrand = torch.sigmoid(self._alpha_net(t.reshape(-1, 1)) + beta_exp.reshape(-1, 1)).view(n, -1, 1)
        sigmoid_integral = 0.5 * (theta - a) * (weights * sigmoid_integrand).sum(dim=1)  # (n, 1)
        dr_dT = b_prime_T + beta_prime_T * sigmoid_integral  # (n, 1)
        # Clamp away from 0 just in case (b'_umnn is already > 0 by softplus)
        log_det_jac_input = torch.log(dr_dT.clamp_min(1e-12)).squeeze(-1)  # (n,)

        return r, log_det_jac_input

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
