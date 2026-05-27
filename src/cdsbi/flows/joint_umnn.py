"""JointUMNNFlow — the R1-only ablation flow for §8.4.

Monotone in θ via a UMNN integrand taking (θ, T) jointly, but with no
architectural constraint on monotonicity in T. log|∂_T r| comes from
autograd, which is exactly the §3.5 failure regime: when autograd's
local Jacobian doesn't enforce positive ∂_T r everywhere, the surrogate
density's mass Z(θ) exceeds 1 and the NF-MLE loss drops below the
information-theoretic floor.

advertised monotonicity_guarantees = {R1}; CDSBIRunner refuses to
construct unless allow_ablation=True.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cdsbi.flows.base import Flow, Guarantee


_NODES_NP, _WEIGHTS_NP = np.polynomial.legendre.leggauss(12)


class JointUMNNFlow(nn.Module, Flow):
    monotonicity_guarantees = frozenset({Guarantee.R1})

    def __init__(self, hidden: int = 16, theta_ref: float = 0.3):
        super().__init__()
        self.hidden = hidden
        self.theta_ref = theta_ref
        # Joint integrand MLP — takes (θ, T) jointly, outputs a scalar to be softplus'd
        self._integrand_mlp = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Constant term b(T) — generic MLP, no monotonicity constraint in T
        self._b_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self._integrand_mlp[-1].weight)
        nn.init.zeros_(self._integrand_mlp[-1].bias)
        nn.init.zeros_(self._b_mlp[-1].weight)
        nn.init.zeros_(self._b_mlp[-1].bias)
        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _integrand(self, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """softplus(MLP([t, T])) — strictly positive in t direction by softplus,
        but no architectural sign-of-derivative guarantee in T."""
        inputs = torch.cat([t, T], dim=-1)
        return F.softplus(self._integrand_mlp(inputs)) + 1e-3

    def _r(self, theta: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Compute r(θ; T) = b(T) + ∫_{theta_ref}^{theta} g(t, T) dt via 12-pt Gauss-Legendre.

        Used twice in forward(): once with detached T for the returned r, and (when
        T has no grad) once with a grad-requiring leaf clone of T so autograd can
        give us ∂r/∂T."""
        n = theta.shape[0]
        a = self.theta_ref
        # Quadrature nodes mapped from [-1, 1] to [theta_ref, theta]; broadcasting
        # turns the (n, 1, 1) × (1, K, 1) product into (n, K, 1).
        u = self._nodes.view(1, -1, 1)
        t = a + 0.5 * (theta.view(n, 1, 1) - a) * (u + 1.0)
        T_broadcast = T.view(n, 1, 1).expand(-1, u.size(1), -1)
        integrand = self._integrand(t.reshape(-1, 1), T_broadcast.reshape(-1, 1)).view(n, -1, 1)
        integral = 0.5 * (theta - a) * (self._weights.view(1, -1, 1) * integrand).sum(dim=1)
        return self._b_mlp(T) + integral

    def forward(
        self, theta: torch.Tensor, context: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert context is not None and context.shape[-1] == 1
        T = context

        # ∂r/∂T comes from autograd. That requires T to be a leaf with
        # requires_grad=True (or a non-leaf in an existing graph). If the caller
        # didn't supply one, we recompute r on a fresh grad-requiring leaf clone
        # of T just for the autograd.grad call; the *returned* r uses the
        # original detached T so no spurious grads leak upstream.
        if T.requires_grad:
            r = self._r(theta, T)
            grad_input = T
            r_for_grad = r
        else:
            r = self._r(theta, T)
            grad_input = T.detach().clone().requires_grad_(True)
            r_for_grad = self._r(theta, grad_input)

        grad_T = torch.autograd.grad(
            r_for_grad.sum(), grad_input, create_graph=self.training, retain_graph=True,
        )[0]
        log_det_jac_input = torch.log(grad_T.abs().clamp_min(1e-12)).squeeze(-1)

        return r, log_det_jac_input

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
