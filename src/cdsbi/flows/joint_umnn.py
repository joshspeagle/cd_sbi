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

    def forward(
        self, theta: torch.Tensor, context: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert context is not None and context.shape[-1] == 1
        T = context
        n = theta.shape[0]
        a = self.theta_ref

        # Quadrature nodes on [theta_ref, theta]
        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)
        b = theta.view(n, 1, 1).expand(-1, u.size(1), -1)
        t = a + 0.5 * (b - a) * (u + 1.0)
        T_exp = T.unsqueeze(1).expand(-1, u.size(1), -1)
        integrand = self._integrand(t.reshape(-1, 1), T_exp.reshape(-1, 1)).view(n, -1, 1)
        weights = self._weights.view(1, -1, 1)
        integral = 0.5 * (theta - a) * (weights * integrand).sum(dim=1)

        # Make sure T is included in the autograd graph (it's the conditioner output
        # — typically passed in detached). To compute ∂r/∂T via autograd we need
        # T to be a leaf with requires_grad=True OR a non-leaf that's part of the
        # graph. We force the latter by including a no-op identity that touches T.
        # If the caller passed T as a leaf with requires_grad=True, that path also
        # works.
        if not T.requires_grad:
            T_grad = T.clone().detach().requires_grad_(True)
            # Recompute the integral with the grad-requiring T to get an accurate
            # ∂_T r through autograd (the path above used the non-grad T).
            T_grad_exp = T_grad.unsqueeze(1).expand(-1, u.size(1), -1)
            integrand_g = self._integrand(t.reshape(-1, 1), T_grad_exp.reshape(-1, 1)).view(n, -1, 1)
            integral_g = 0.5 * (theta - a) * (weights * integrand_g).sum(dim=1)
            b_g = self._b_mlp(T_grad)
            r_for_grad = b_g + integral_g
            grad_T = torch.autograd.grad(
                r_for_grad.sum(), T_grad, create_graph=self.training, retain_graph=True,
            )[0]
        else:
            b_T = self._b_mlp(T)
            r_for_grad = b_T + integral
            grad_T = torch.autograd.grad(
                r_for_grad.sum(), T, create_graph=self.training, retain_graph=True,
            )[0]

        b_T_eval = self._b_mlp(T.detach() if not T.requires_grad else T)
        r = b_T_eval + integral
        log_det_jac_input = torch.log(grad_T.abs().clamp_min(1e-12)).squeeze(-1)

        return r, log_det_jac_input

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
