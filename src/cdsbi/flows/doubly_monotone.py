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


def _tanh_mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    """Standard depth-d tanh MLP.

    Codebase convention: depth=2 (matches UMNNBlock, the §8.4 reference
    implementation, and the default in NRE/LF2I config groups). Exposed as
    a parameter on the v3 flow classes so depth can be swept as a separate
    architectural axis without touching the hidden width.
    """
    assert depth >= 1, f"depth must be >= 1 (got {depth})"
    layers: list = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class _MonotoneScalarUMNN(nn.Module):
    """A 1D monotone-increasing function of T via T → bias + ∫_{T_ref}^T softplus(MLP(t)) dt.

    Self-contained quadrature; does not depend on v0's UMNNBlock to keep the
    integral signature clean (we use both the value and the derivative).
    T_ref is the integration baseline — choose it near the typical T value
    so the `bias` parameter has the interpretation of "value at typical T"
    and the integral represents "deviation from typical." T_ref=0 (the
    earlier default) integrated from 0 to T, making the integral large for
    typical T values and the bias dominated; T_ref=5.0 (the §8.4 reference
    implementation default) better matches optimization scales.
    """

    def __init__(self, hidden: int = 16, depth: int = 2, T_ref: float = 5.0):
        super().__init__()
        self.T_ref = T_ref
        self.mlp = _tanh_mlp(in_dim=1, hidden=hidden, out_dim=1, depth=depth)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _integrand(self, t: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.mlp(t)) + 1e-3  # strictly positive

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        """bias + ∫_{T_ref}^T softplus(MLP(t)) dt. Integral can be negative
        if T < T_ref (the integrand is positive, but the interval reverses)."""
        n = T.shape[0]
        a = self.T_ref
        # Map [-1, 1] nodes to [T_ref, T]
        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)
        T_exp = T.view(n, 1, 1).expand(-1, u.size(1), -1)
        t = a + 0.5 * (T_exp - a) * (u + 1.0)
        integrand = self._integrand(t)
        weights = self._weights.view(1, -1, 1)
        integral = 0.5 * (T - a) * (weights * integrand).sum(dim=1)
        return self.bias + integral

    def derivative(self, T: torch.Tensor) -> torch.Tensor:
        """f'(T) = softplus(MLP(T))."""
        return self._integrand(T)


class DoublyMonotoneUMNN(nn.Module, Flow):
    monotonicity_guarantees = frozenset({Guarantee.R1, Guarantee.R2})

    def __init__(self, hidden: int = 16, theta_ref: float = 0.3, depth: int = 2):
        super().__init__()
        self.hidden = hidden
        self.theta_ref = theta_ref
        self.depth = depth
        # b_umnn(T): scalar monotone-increasing function of T
        self._b_umnn = _MonotoneScalarUMNN(hidden=hidden, depth=depth)
        # β_umnn(T): scalar monotone-increasing function of T (parameter inside the integrand)
        self._beta_umnn = _MonotoneScalarUMNN(hidden=hidden, depth=depth)
        # α(t): unconstrained MLP of t (the θ variable). Depth defaults to
        # 2 per the codebase convention; configurable for capacity sweeps.
        # A 1-hidden-layer α-net was the earlier bug (caught by diffing
        # against the §8.4 reference implementation): the α-net needs
        # enough capacity to fit the χ²-CDF shape that ∂_θ r* implies.
        self._alpha_net = _tanh_mlp(in_dim=1, hidden=hidden, out_dim=1, depth=depth)
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
