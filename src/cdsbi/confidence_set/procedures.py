"""ConfidenceProcedure subtypes — what every method's TrainedModel.procedure is.

Each subtype produces a ConfidenceSet via its own mechanism:
- PivotBased: chi-square inversion of ‖r‖² ≤ χ²_{d, α}
- CriticalValue: {θ : T(θ, X) ≤ c_α(θ)} (LF2I)
- PosteriorBased: equal-tailed credible interval on sampled posterior (NPE)
- LikelihoodBased: Wilks-style likelihood-ratio inversion (NLE)
- RatioBased: ratio thresholding (NRE)
"""
from __future__ import annotations

from typing import Callable, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
from scipy.stats import chi2

from cdsbi.confidence_set.datatypes import ConfidenceSet
from cdsbi.confidence_set.root_find import bisect_1d


def _theta_to_row_tensor(theta_value, dtype, device, d_theta: int) -> torch.Tensor:
    """Normalise a scalar / sequence / tensor θ_0 into a (1, d_theta) row.

    Handles Python float / int, numpy scalar / 0-d tensor (np.ndim == 0),
    Python list / tuple, 1-d ndarray, 1-d torch.Tensor (CPU or CUDA), 0-d
    torch.Tensor.
    """
    if isinstance(theta_value, torch.Tensor):
        # Avoid numpy detour for CUDA tensors (np.asarray raises on them).
        t = theta_value.detach().to(dtype=dtype, device=device).reshape(-1)
    else:
        arr = np.atleast_1d(np.asarray(theta_value, dtype=np.float64)).reshape(-1)
        t = torch.from_numpy(arr).to(dtype=dtype, device=device)
    assert t.shape == (d_theta,), (
        f"theta_0 has shape {tuple(t.shape)}, expected ({d_theta},)"
    )
    return t.view(1, d_theta)


def _vectorized_bisect_1d(
    f_batch,
    a: torch.Tensor,
    b: torch.Tensor,
    f_sign_at_a: int,
    n_iters: int = 40,
) -> torch.Tensor:
    """Vectorized 1D bisection across a batch.

    f_batch(theta_vals: (B,)) -> (B,) scalar values of f at theta_vals per
    batch element. The bisection assumes a sign change on each interval
    [a[i], b[i]] and that f_sign_at_a is the (constant) sign of f(a) across
    the batch (+1 if f(a) > 0, -1 if f(a) < 0). Returns the root (B,) tensor.

    Caller is responsible for verifying f really changes sign on the
    interval; if not, the returned value is the midpoint of the bracket.
    """
    for _ in range(n_iters):
        m = 0.5 * (a + b)
        f_m = f_batch(m)
        if f_sign_at_a > 0:
            # f > 0 at a, f < 0 at b. f(m) > 0 → root is in [m, b].
            move_a = f_m > 0
        else:
            # f < 0 at a, f > 0 at b. f(m) < 0 → root is in [m, b].
            move_a = f_m < 0
        a = torch.where(move_a, m, a)
        b = torch.where(move_a, b, m)
    return 0.5 * (a + b)


class _BatchCache:
    """Identity-keyed cache for α-invariant intermediates.

    Coverage / SetSize iterate (θ_0, α) with x_obs_batch fixed across the α
    inner loop. Many procedure-level intermediates (posterior samples, grid
    log-likelihoods, test-statistic evaluations) depend only on x_obs_batch
    (and sometimes θ_0), not α — so caching them per (θ_0, x_obs_batch) gives
    a 4× speedup over the 4 default α values.

    Cache invalidation: on key mismatch, the new value replaces the old. No
    LRU, no manual flush — caller passes a fresh `key` each time and the
    cache automatically tracks the most recent (θ_0, x_obs_batch).
    """

    def __init__(self):
        self._key = None
        self._value = None

    def get_or_compute(self, key, compute_fn):
        if self._key == key:
            return self._value
        v = compute_fn()
        self._key = key
        self._value = v
        return v

    def reset(self):
        self._key = None
        self._value = None


def _ray_sample_set_boundary(
    x_obs: torch.Tensor,
    inside_fn,
    theta_range: tuple,
    d: int,
    n_rays: int = 200,
    n_grid_per_dim: int = 9,
    n_newton: int = 8,
) -> tuple:
    """Multivariate confidence-set boundary via line-sampling.

    Args:
        x_obs: shape (1, d_x).
        inside_fn: callable (theta: (B, d), x_obs: (B, d_x)) -> (B,) tensor.
            inside_fn(θ; X) ≤ 0 iff θ ∈ α-set.
        theta_range: (lo, hi) bounds of the prior box.
        d: dimension of θ.
        n_rays: number of random unit directions on S^{d-1} to sample.
        n_grid_per_dim: coarse-grid resolution per coord for center seed.
        n_newton: L-BFGS iterations for center refinement.

    Returns (contains_fn, boundary_repr): caller wraps in a ConfidenceSet with
    the appropriate α. boundary_repr has shape (n_rays, d), or (0, d) when the
    set is empty (center itself violates inside_fn ≤ 0).

    Assumes the α-set is radially convex around the center — same precondition
    as PivotBasedProcedure._confidence_set_ray_sampled.
    """
    dtype = x_obs.dtype
    device = x_obs.device
    lo, hi = theta_range

    # 1) Coarse grid argmin of inside_fn.
    axes = [torch.linspace(lo, hi, n_grid_per_dim, dtype=dtype, device=device)
            for _ in range(d)]
    mesh = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).view(-1, d)
    with torch.no_grad():
        x_batch = x_obs.expand(mesh.shape[0], -1)
        vals = inside_fn(mesh, x_batch)
        best = int(torch.argmin(vals).item())
    center = mesh[best].clone().detach().requires_grad_(True)

    # 2) L-BFGS refinement.
    opt = torch.optim.LBFGS([center], lr=0.5, max_iter=n_newton)

    def _closure():
        opt.zero_grad()
        loss = inside_fn(center.unsqueeze(0), x_obs).sum()
        loss.backward()
        return loss

    try:
        opt.step(_closure)
    except RuntimeError:
        # inside_fn may not be differentiable w.r.t. θ (e.g. critical-value
        # MLP returns a constant); fall back to the grid argmin.
        pass
    center = center.detach()

    # 3) Empty-set short-circuit.
    with torch.no_grad():
        v_c = float(inside_fn(center.unsqueeze(0), x_obs).item())
    if v_c > 0:
        empty_boundary = torch.empty((0, d), dtype=dtype, device=device)
        return (lambda _theta_val: False), empty_boundary

    # 4) Sample n_rays unit directions and bisect along each.
    u = torch.randn(n_rays, d, dtype=dtype, device=device)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    t_lo = torch.zeros(n_rays, dtype=dtype, device=device)
    t_hi = torch.full((n_rays,), 2.0 * (hi - lo), dtype=dtype, device=device)
    x_batch = x_obs.expand(n_rays, -1)
    for _ in range(40):
        m = 0.5 * (t_lo + t_hi)
        theta_m = center.unsqueeze(0) + m.unsqueeze(-1) * u
        v = inside_fn(theta_m, x_batch)
        if v.device != device:
            v = v.to(device)
        outside = v > 0
        t_hi = torch.where(outside, m, t_hi)
        t_lo = torch.where(outside, t_lo, m)
    t = 0.5 * (t_lo + t_hi)
    boundary = center.unsqueeze(0) + t.unsqueeze(-1) * u

    def _contains(theta_val) -> bool:
        theta_t = _theta_to_row_tensor(theta_val, dtype, device, d)
        v = inside_fn(theta_t, x_obs)
        return bool(float(v.item()) <= 0)

    return _contains, boundary


@runtime_checkable
class ConfidenceProcedure(Protocol):
    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet: ...


class PivotBasedProcedure:
    """CDSBI: 1D pivot inverted via chi-square."""

    def __init__(self, pivot_fn: Callable, d_theta: int, theta_range: tuple = (-20.0, 20.0)):
        self.pivot_fn = pivot_fn
        self.d_theta = d_theta
        self.theta_range = theta_range

    def pivot(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.pivot_fn(theta, x)

    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        thresh = float(chi2.ppf(alpha, df=self.d_theta))
        if self.d_theta == 1:
            return self._confidence_set_1d(x_obs, alpha, thresh)
        return self._confidence_set_ray_sampled(x_obs, alpha, thresh, n_rays=200)

    def _confidence_set_1d(self, x_obs, alpha, thresh):
        # Existing v0 1D bisection body — chi-square inversion via
        # 1D bisection of f(θ) = r(θ)² − thresh on either side of the
        # center where r=0.
        def f(theta_val: float) -> float:
            theta = torch.tensor([[theta_val]], dtype=x_obs.dtype, device=x_obs.device)
            r = self.pivot_fn(theta, x_obs)
            return (r.pow(2).sum().item() - thresh)

        def r_only(theta_val: float) -> float:
            theta = torch.tensor([[theta_val]], dtype=x_obs.dtype, device=x_obs.device)
            return self.pivot_fn(theta, x_obs).item()

        lo, hi = self.theta_range
        center = bisect_1d(r_only, lo, hi, tol=1e-4)
        left = bisect_1d(f, lo, center, tol=1e-4)
        right = bisect_1d(f, center, hi, tol=1e-4)

        def contains(theta_val) -> bool:
            v = float(theta_val) if not hasattr(theta_val, "__len__") else float(theta_val[0])
            return left <= v <= right

        return ConfidenceSet(
            contains=contains, boundary_repr=torch.tensor([left, right]), alpha=alpha,
        )

    def _find_center(self, x_obs, n_grid_per_dim: int = 9, n_newton: int = 8):
        """Find a θ_center inside the α-set with ||r||² near 0 (or its minimum).

        For an oracle location-form pivot (r = θ - X), the min is at θ = X_obs.
        For a trained flow, r(X_obs; X_obs) is generally non-zero, so we:
          (1) evaluate ||r(θ; X_obs)||² on a coarse d-dim grid over theta_range
              (n_grid_per_dim^d points; cheap in d=2-3),
          (2) pick the argmin as a starting point,
          (3) refine by a few L-BFGS steps.
        Returns a (d,) tensor.
        """
        dtype = x_obs.dtype
        device = x_obs.device
        d = self.d_theta
        lo, hi = self.theta_range
        # Coarse grid argmin
        axes = [torch.linspace(lo, hi, n_grid_per_dim, dtype=dtype, device=device)
                for _ in range(d)]
        mesh = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).view(-1, d)
        with torch.no_grad():
            x_batch = x_obs.expand(mesh.shape[0], -1)
            r = self.pivot_fn(mesh, x_batch)
            sq = r.pow(2).sum(dim=-1)
            best = int(torch.argmin(sq).item())
        center = mesh[best].clone().detach().requires_grad_(True)
        opt = torch.optim.LBFGS([center], lr=0.5, max_iter=n_newton)

        def _closure():
            opt.zero_grad()
            r = self.pivot_fn(center.unsqueeze(0), x_obs)
            loss = r.pow(2).sum()
            loss.backward()
            return loss

        opt.step(_closure)
        return center.detach()

    def _confidence_set_ray_sampled(self, x_obs, alpha, thresh, n_rays: int):
        """Multivariate boundary via line-sampling.

        Finds a center via _find_center (argmin of ||r||²), then for each of
        `n_rays` random unit directions u_i ∈ S^{d-1} bisects along the ray
        to find t_i where ||r(center + t · u_i; X)||² = thresh.

        Note: this assumes f(t) = ||r(center + t·u; X)||² − thresh is monotone
        in t along each ray near the boundary, i.e. the α-set is radially
        convex around the center. True for all ellipsoids and well-behaved
        pivots; pathological non-star-shaped sets would need a different
        boundary representation.
        """
        dtype = x_obs.dtype
        device = x_obs.device
        d = self.d_theta
        center = self._find_center(x_obs)  # (d,)

        # If center is itself outside the α-set, signal an empty set.
        with torch.no_grad():
            r_c = self.pivot_fn(center.unsqueeze(0), x_obs)
            sq_c = float(r_c.pow(2).sum().item())
        if sq_c > thresh:
            empty_boundary = torch.empty((0, d), dtype=dtype, device=device)

            def _contains_empty(theta_val) -> bool:
                return False
            return ConfidenceSet(
                contains=_contains_empty, boundary_repr=empty_boundary, alpha=alpha,
            )

        # Sample n_rays unit directions uniformly on S^{d-1}.
        u = torch.randn(n_rays, d, dtype=dtype, device=device)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # 1D bisection along each ray. f(t=0) < 0 (center is inside),
        # f(t=t_max) > 0 (far enough is outside).
        lo, hi = self.theta_range
        t_lo = torch.zeros(n_rays, dtype=dtype, device=device)
        t_hi = torch.full((n_rays,), 2.0 * (hi - lo), dtype=dtype, device=device)
        x_batch = x_obs.expand(n_rays, -1)
        for _ in range(40):
            m = 0.5 * (t_lo + t_hi)
            theta_m = center.unsqueeze(0) + m.unsqueeze(-1) * u
            r = self.pivot_fn(theta_m, x_batch)
            # pivot_fn may live on a different device than x_obs (e.g. a CUDA
            # flow being queried with a CPU x_obs from a numpy-rng simulator).
            # Bring r back to the procedure's device so torch.where works.
            if r.device != device:
                r = r.to(device)
            f = r.pow(2).sum(dim=-1) - thresh
            outside = f > 0
            t_hi = torch.where(outside, m, t_hi)
            t_lo = torch.where(outside, t_lo, m)
        t = 0.5 * (t_lo + t_hi)
        boundary = center.unsqueeze(0) + t.unsqueeze(-1) * u

        def contains(theta_val) -> bool:
            # Use the same normalisation helper as the rest of this module so
            # CUDA-tensor θ inputs don't trip on np.asarray.
            theta_t = _theta_to_row_tensor(theta_val, dtype, device, d)
            r = self.pivot_fn(theta_t, x_obs)
            return bool((r.pow(2).sum().item() <= thresh))

        return ConfidenceSet(contains=contains, boundary_repr=boundary, alpha=alpha)

    def contains_batch(
        self, theta_0_value, x_obs_batch: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """Vectorized containment test: returns a bool tensor of shape (B,)
        indicating whether `theta_0_value` is inside the α-confidence set
        for each X_obs in the batch.

        O(1) flow forward (one batched call) — used by Coverage.
        """
        thresh = float(chi2.ppf(alpha, df=self.d_theta))
        B = x_obs_batch.shape[0]
        theta_t = _theta_to_row_tensor(
            theta_0_value, x_obs_batch.dtype, x_obs_batch.device, self.d_theta,
        ).expand(B, -1)
        r = self.pivot_fn(theta_t, x_obs_batch)  # (B, d_theta)
        r_sq = r.pow(2).sum(dim=-1)               # (B,)
        return r_sq <= thresh

    def confidence_set_batch(
        self, x_obs_batch: torch.Tensor, alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized chi-square inversion across a batch of X_obs.

        Returns (left, right): each shape (B,). Only implemented for d_theta=1.
        For each x_obs, the confidence interval is {θ : r(θ, x_obs)² ≤ chi²_{1, α}}.

        Uses three fixed-iteration (40 iters) vectorized bisections in parallel
        across the batch. At 40 iters over a 40-wide default range, tol ≈ 3.6e-11.
        """
        assert self.d_theta == 1, "confidence_set_batch only supports d_theta=1 in v0"
        thresh = float(chi2.ppf(alpha, df=1))
        B = x_obs_batch.shape[0]
        lo, hi = self.theta_range
        dtype = x_obs_batch.dtype

        # Probe pivot_fn once to learn the model's device — the closure may move
        # inputs to a device other than x_obs_batch.device, and all working
        # tensors (a, b, m, center) below need to live on the same device as the
        # probe output for the torch.where reductions to work.
        probe = self.pivot_fn(
            torch.zeros(1, 1, dtype=dtype, device=x_obs_batch.device),
            x_obs_batch[:1],
        )
        device = probe.device
        if x_obs_batch.device != device:
            x_obs_batch = x_obs_batch.to(device)

        # Vectorized bisection #1: find center where r=0 for each x_obs.
        # r is monotone increasing in θ (architectural guarantee), so r(lo)<0<r(hi).
        # r(m) >= 0  →  root is in [a, m]  →  b = m
        # r(m) <  0  →  root is in [m, b]  →  a = m
        a = torch.full((B,), lo, dtype=dtype, device=device)
        b = torch.full((B,), hi, dtype=dtype, device=device)
        for _ in range(40):
            m = 0.5 * (a + b)
            r_m = self.pivot_fn(m.unsqueeze(-1), x_obs_batch).squeeze(-1)
            root_le_m = r_m >= 0  # root is at or left of m
            b = torch.where(root_le_m, m, b)
            a = torch.where(root_le_m, a, m)
        center = 0.5 * (a + b)

        # Helper: f(θ) = r(θ)² − thresh for a batch of θ values
        def f_batch(theta_vec: torch.Tensor) -> torch.Tensor:
            r = self.pivot_fn(theta_vec.unsqueeze(-1), x_obs_batch).squeeze(-1)
            return r.pow(2) - thresh

        # Vectorized bisection #2: find left root of f on [lo, center].
        # On this interval f is decreasing (r goes from large negative → 0):
        # f(lo) > 0, f(center) ≤ 0.
        # f(m) > 0  →  root is in [m, center]  →  a = m
        # f(m) ≤ 0  →  root is in [lo, m]      →  b = m
        a = torch.full((B,), lo, dtype=dtype, device=device)
        b = center.clone()
        for _ in range(40):
            m = 0.5 * (a + b)
            f_m = f_batch(m)
            root_ge_m = f_m > 0  # root is at or right of m (f still positive)
            a = torch.where(root_ge_m, m, a)
            b = torch.where(root_ge_m, b, m)
        left = 0.5 * (a + b)

        # Vectorized bisection #3: find right root of f on [center, hi].
        # On this interval f is increasing (r goes from 0 → large positive):
        # f(center) ≤ 0, f(hi) > 0.
        # f(m) < 0  →  root is in [m, hi]     →  a = m
        # f(m) ≥ 0  →  root is in [center, m]  →  b = m
        a = center.clone()
        b = torch.full((B,), hi, dtype=dtype, device=device)
        for _ in range(40):
            m = 0.5 * (a + b)
            f_m = f_batch(m)
            root_ge_m = f_m < 0  # f still negative: root is to the right
            a = torch.where(root_ge_m, m, a)
            b = torch.where(root_ge_m, b, m)
        right = 0.5 * (a + b)

        return left, right


class CriticalValueProcedure:
    """LF2I: {θ : T(θ, X) ≤ c_α(θ)}."""

    def __init__(
        self,
        test_stat_fn: Callable,
        critical_value_fn: Callable,
        d_theta: int,
        theta_range: tuple = (-20.0, 20.0),
    ):
        self.test_stat_fn = test_stat_fn
        self.critical_value_fn = critical_value_fn
        self.d_theta = d_theta
        self.theta_range = theta_range
        self._t_obs_cache = _BatchCache()
        self._t_grid_cache = _BatchCache()
        self._probed_device = None

    def test_statistic(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.test_stat_fn(theta, x)

    def critical_value(self, theta: torch.Tensor, alpha: float) -> torch.Tensor:
        return self.critical_value_fn(theta, alpha)

    def contains_batch(
        self, theta_0_value, x_obs_batch: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """Vectorized containment test: returns a bool tensor of shape (B,)
        indicating whether `theta_0_value` is inside the α-confidence set
        for each X_obs in the batch.

        For CriticalValueProcedure the set is {θ : T(θ; X_obs) ≤ c_α(θ)};
        containing θ_0 reduces to T(θ_0; X_obs_i) ≤ c_α(θ_0) for each i,
        and c_α(θ_0) is the same scalar for every element of the batch
        (we evaluate the critical-value MLP once).
        """
        B = x_obs_batch.shape[0]
        # Probe the test stat once to learn the device the closures operate on;
        # cache the result so subsequent Coverage α-loop calls skip the extra
        # forward (matters when test_stat_fn is expensive, e.g. BFF marginal).
        if self._probed_device is None:
            theta_probe = _theta_to_row_tensor(
                theta_0_value, x_obs_batch.dtype, x_obs_batch.device, self.d_theta,
            )
            probe = self.test_stat_fn(theta_probe, x_obs_batch[:1])
            self._probed_device = probe.device
        device = self._probed_device
        if x_obs_batch.device != device:
            x_obs_batch = x_obs_batch.to(device)
        theta_t = _theta_to_row_tensor(
            theta_0_value, x_obs_batch.dtype, device, self.d_theta,
        ).expand(B, -1)
        # t_obs depends on (θ_0, X_obs); α only enters via c_α(θ_0).
        # Cache by (id(x_obs_batch), θ_0 repr) so Coverage's α loop hits.
        theta_0_repr = tuple(np.atleast_1d(
            np.asarray(theta_0_value if not isinstance(theta_0_value, torch.Tensor)
                       else theta_0_value.detach().cpu(), dtype=np.float64)
        ).reshape(-1).tolist())
        cache_key = ("t_obs", id(x_obs_batch), theta_0_repr)

        def _compute_t_obs():
            t = self.test_stat_fn(theta_t, x_obs_batch)
            if t.ndim > 1:
                t = t.squeeze(-1)
            return t

        t_obs = self._t_obs_cache.get_or_compute(cache_key, _compute_t_obs)
        # Critical value at the (single) θ_0 — scalar across the batch.
        c_val = self.critical_value_fn(theta_t[:1], alpha)
        if c_val.ndim == 0:
            c_scalar = c_val
        else:
            c_scalar = c_val[0]
        return t_obs <= c_scalar

    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        if self.d_theta == 1:
            return self._confidence_set_1d(x_obs, alpha)
        return self._confidence_set_d_gt_1(x_obs, alpha)

    def _confidence_set_1d(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        def f(theta_val: float) -> float:
            theta = torch.tensor([[theta_val]], dtype=x_obs.dtype, device=x_obs.device)
            t = self.test_stat_fn(theta, x_obs).item()
            c = self.critical_value_fn(theta, alpha).item()
            return t - c  # negative ⇒ in set; positive ⇒ outside

        lo, hi = self.theta_range
        grid = torch.linspace(lo, hi, 200).tolist()
        signs = [f(t) <= 0 for t in grid]
        try:
            i_first = signs.index(True)
            i_last = len(signs) - 1 - signs[::-1].index(True)
        except ValueError:
            return ConfidenceSet(
                contains=lambda th: False, boundary_repr=torch.tensor([0.0, 0.0]), alpha=alpha
            )
        left_lo, left_hi = grid[max(i_first - 1, 0)], grid[i_first]
        right_lo, right_hi = grid[i_last], grid[min(i_last + 1, len(grid) - 1)]
        left = bisect_1d(f, left_lo, left_hi, tol=1e-4) if i_first > 0 else grid[0]
        right = bisect_1d(f, right_lo, right_hi, tol=1e-4) if i_last < len(grid) - 1 else grid[-1]

        def contains(theta_val: float) -> bool:
            return left <= theta_val <= right

        return ConfidenceSet(
            contains=contains, boundary_repr=torch.tensor([left, right]), alpha=alpha
        )

    def _confidence_set_d_gt_1(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        def inside_fn(theta, x):
            t = self.test_stat_fn(theta, x)
            c = self.critical_value_fn(theta, alpha)
            if t.ndim > 1:
                t = t.squeeze(-1)
            if c.ndim > 1:
                c = c.squeeze(-1)
            return t - c

        contains_fn, boundary = _ray_sample_set_boundary(
            x_obs, inside_fn, self.theta_range, self.d_theta, n_rays=200,
        )
        return ConfidenceSet(contains=contains_fn, boundary_repr=boundary, alpha=alpha)

    def confidence_set_batch(
        self, x_obs_batch: torch.Tensor, alpha: float, n_grid: int = 200,
    ) -> tuple:
        """Vectorized 1D LF2I confidence-set boundaries across a batch of X_obs.
        Returns (left, right), each shape (B,). 1D only in v1.
        """
        assert self.d_theta == 1, "confidence_set_batch is 1D only in v1"
        # Probe device once and cache (shared with contains_batch's probe).
        if self._probed_device is None:
            probe = self.test_stat_fn(
                torch.zeros((1, 1), dtype=x_obs_batch.dtype, device=x_obs_batch.device),
                x_obs_batch[:1],
            )
            self._probed_device = probe.device
        device = self._probed_device
        if x_obs_batch.device != device:
            x_obs_batch = x_obs_batch.to(device)
        B = x_obs_batch.shape[0]
        lo, hi = self.theta_range

        # Coarse-grid eval of T — α-invariant, cache it.
        def _compute_t_grid():
            theta_grid_ = torch.linspace(
                lo, hi, n_grid, device=device, dtype=x_obs_batch.dtype,
            ).view(-1, 1)
            theta_grid_exp = theta_grid_.unsqueeze(0).expand(B, -1, -1).reshape(-1, 1)
            x_obs_exp = x_obs_batch.unsqueeze(1).expand(-1, n_grid, -1).reshape(
                -1, x_obs_batch.shape[-1]
            )
            t_g = self.test_stat_fn(theta_grid_exp, x_obs_exp).view(B, n_grid)
            return theta_grid_, t_g

        cache_key = ("t_grid", id(x_obs_batch), int(n_grid))
        theta_grid, t_grid = self._t_grid_cache.get_or_compute(
            cache_key, _compute_t_grid,
        )
        # c_α(θ) only depends on θ → evaluate once, broadcast.
        c_grid = self.critical_value_fn(theta_grid, alpha)
        if c_grid.ndim == 2:
            c_grid = c_grid.squeeze(-1)
        f_grid = t_grid - c_grid.unsqueeze(0)  # (B, n_grid)
        center_idx = f_grid.argmin(dim=1)  # most-inside grid pt
        center_theta = theta_grid.squeeze(-1)[center_idx]

        def f_batch(theta_vals: torch.Tensor) -> torch.Tensor:
            theta_b = theta_vals.unsqueeze(-1)
            t = self.test_stat_fn(theta_b, x_obs_batch).squeeze(-1)
            c = self.critical_value_fn(theta_b, alpha)
            if c.ndim > 1:
                c = c.squeeze(-1)
            return t - c

        a = torch.full((B,), lo, dtype=x_obs_batch.dtype, device=device)
        b = center_theta.clone()
        left = _vectorized_bisect_1d(f_batch, a, b, f_sign_at_a=+1)
        a = center_theta.clone()
        b = torch.full((B,), hi, dtype=x_obs_batch.dtype, device=device)
        right = _vectorized_bisect_1d(f_batch, a, b, f_sign_at_a=-1)
        return left, right


class PosteriorBasedProcedure:
    """NPE: equal-tailed credible interval via posterior samples.

    TODO(v1+): switch to a true kernel-based HPD when skewed posteriors enter
    the picture — equal-tailed ≠ HPD for asymmetric distributions.
    """

    def __init__(
        self,
        sample_fn: Callable,
        d_theta: int,
        sample_batched_fn: Callable = None,
    ):
        self.sample_fn = sample_fn
        self.d_theta = d_theta
        # Optional vectorized sampler: takes (x_obs_batch (B, d_x), n) and
        # returns samples shape (n, B, d_theta). Lets contains_batch use a
        # single batched posterior.sample call instead of B serial ones.
        self.sample_batched_fn = sample_batched_fn
        self._sample_cache = _BatchCache()

    def posterior_samples(self, x_obs: torch.Tensor, n: int = 10_000) -> torch.Tensor:
        return self.sample_fn(x_obs, n)

    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        from cdsbi.confidence_set.equal_tailed import equal_tailed_1d
        n_samples = 10_000
        samples = self.sample_fn(x_obs, n_samples)
        if self.d_theta == 1:
            return equal_tailed_1d(samples.flatten(), alpha=alpha)
        return self._confidence_set_mahalanobis(samples, alpha)

    def _confidence_set_mahalanobis(
        self, samples: torch.Tensor, alpha: float, n_boundary: int = 200,
    ) -> ConfidenceSet:
        """Empirical Mahalanobis credible region from posterior samples.

        Returns a ConfidenceSet whose contains() uses the Mahalanobis
        inequality and whose boundary_repr is n_boundary points sampled
        uniformly on the ellipsoid surface in whitened space.

        Caveat: this approximates the α-HPD as a Mahalanobis ellipse — exact
        for Gaussian posteriors (matches the analytic HPD), correlation-aware
        for moderately non-Gaussian ones, but biased for strongly skewed or
        multi-modal posteriors. A true KDE-based HPD lands in v1+ if/when a
        non-Gaussian target arrives.
        """
        if samples.ndim != 2:
            samples = samples.view(-1, self.d_theta)
        device = samples.device
        dtype = samples.dtype
        d = self.d_theta

        # Empirical moments. torch.cov expects (d, N) input.
        mu = samples.mean(dim=0)  # (d,)
        # Cov-stabilize with a small ridge to keep Cholesky well-conditioned.
        cov = torch.cov(samples.T) + 1e-6 * torch.eye(d, dtype=dtype, device=device)
        L = torch.linalg.cholesky(cov)
        L_inv = torch.linalg.inv(L)

        # Mahalanobis-squared per sample: ||L^{-1} (θ - μ)||².
        delta = samples - mu
        z = delta @ L_inv.T
        sq = z.pow(2).sum(dim=-1)
        thresh = float(torch.quantile(sq, alpha).item())

        # Boundary: K points on the ellipsoid surface in whitened space, mapped
        # back via L. {μ + √thresh · L u_i : u_i ∈ S^{d-1}}.
        u = torch.randn(n_boundary, d, dtype=dtype, device=device)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        radius = thresh ** 0.5
        boundary = mu + (radius * u) @ L.T

        def contains(theta_val) -> bool:
            theta_t = _theta_to_row_tensor(theta_val, dtype, device, d)
            delta = (theta_t - mu).view(-1)
            z_val = L_inv @ delta
            return bool(float(z_val.pow(2).sum().item()) <= thresh)

        return ConfidenceSet(contains=contains, boundary_repr=boundary, alpha=alpha)

    def contains_batch(
        self, theta_0_value, x_obs_batch: torch.Tensor, alpha: float,
        n_samples: int = 10_000,
    ) -> torch.Tensor:
        """Vectorized containment for the equal-tailed posterior interval.

        Requires sample_batched_fn; otherwise falls back to a per-X_obs loop
        through the scalar sample_fn (no real speedup — Coverage would do
        this anyway).
        """
        assert self.d_theta == 1, "contains_batch only supports d_theta=1 in v0"
        B = x_obs_batch.shape[0]
        if self.sample_batched_fn is None:
            results = []
            for i in range(B):
                cs = self.confidence_set(x_obs_batch[i : i + 1], alpha=alpha)
                results.append(cs.contains(theta_0_value))
            return torch.tensor(results)
        cache_key = ("samples", id(x_obs_batch), int(n_samples))
        samples = self._sample_cache.get_or_compute(
            cache_key,
            lambda: self.sample_batched_fn(x_obs_batch, n_samples),
        )  # (n, B, d_θ)
        if samples.ndim == 2:
            # (n, B): treat as d_θ=1 with implicit last dim.
            samples = samples.unsqueeze(-1)
        # Per-X_obs equal-tailed quantiles along the n axis.
        tail = (1.0 - alpha) / 2.0
        lo = torch.quantile(samples, tail, dim=0).squeeze(-1)         # (B,)
        hi = torch.quantile(samples, 1.0 - tail, dim=0).squeeze(-1)  # (B,)
        theta_0_t = torch.tensor(float(theta_0_value), device=lo.device, dtype=lo.dtype)
        return (lo <= theta_0_t) & (theta_0_t <= hi)

    def confidence_set_batch(
        self, x_obs_batch: torch.Tensor, alpha: float, n_samples: int = 10_000,
    ) -> tuple:
        """Vectorized 1D equal-tailed posterior interval boundaries across a
        batch of X_obs. Returns (left, right), each shape (B,). Uses
        sample_batched_fn when available; falls back to a per-X_obs sample_fn
        loop otherwise (no real speedup vs the slow path in that case).
        """
        assert self.d_theta == 1, "confidence_set_batch is 1D only in v1"
        B = x_obs_batch.shape[0]
        tail = (1.0 - alpha) / 2.0
        if self.sample_batched_fn is None:
            lo_list = torch.empty(B, dtype=x_obs_batch.dtype)
            hi_list = torch.empty(B, dtype=x_obs_batch.dtype)
            for i in range(B):
                samples = self.sample_fn(x_obs_batch[i : i + 1], n_samples).flatten()
                lo_list[i] = torch.quantile(samples, tail)
                hi_list[i] = torch.quantile(samples, 1.0 - tail)
            return lo_list, hi_list
        cache_key = ("samples", id(x_obs_batch), int(n_samples))
        samples = self._sample_cache.get_or_compute(
            cache_key,
            lambda: self.sample_batched_fn(x_obs_batch, n_samples),
        )  # (n, B, d_theta)
        if samples.ndim == 2:
            samples = samples.unsqueeze(-1)
        # Per-X_obs quantiles along the sample axis.
        lo_q = torch.quantile(samples.squeeze(-1), tail, dim=0)
        hi_q = torch.quantile(samples.squeeze(-1), 1.0 - tail, dim=0)
        return lo_q, hi_q


class LikelihoodBasedProcedure:
    """NLE: Wilks-style likelihood-ratio confidence set."""

    def __init__(self, log_likelihood_fn: Callable, d_theta: int, theta_range: tuple = (-20.0, 20.0)):
        self.log_likelihood_fn = log_likelihood_fn
        self.d_theta = d_theta
        self.theta_range = theta_range
        self._ll_grid_cache = _BatchCache()

    def log_likelihood(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.log_likelihood_fn(theta, x)

    def contains_batch(
        self, theta_0_value, x_obs_batch: torch.Tensor, alpha: float,
        n_grid: int = 200,
    ) -> torch.Tensor:
        """Vectorized containment for the Wilks LR set:
        {θ : 2(ll_max(X_obs) − ll(θ; X_obs)) ≤ χ²_{d, α}}.

        One batched log_likelihood call over (B × n_grid) (θ, X) pairs gives
        ll_max(X_obs) per element; one more batched call at (θ_0, X_obs_i) gives
        ll(θ_0; X_obs_i). Containment is the elementwise comparison.

        Matches the slow path's grid resolution exactly; bisect refinement of
        the *endpoints* (in `confidence_set`) doesn't affect containment of
        an interior point, only the boundary value.
        """
        thresh = float(chi2.ppf(alpha, df=self.d_theta))
        # Probe device.
        theta_probe = _theta_to_row_tensor(
            theta_0_value, x_obs_batch.dtype, x_obs_batch.device, self.d_theta,
        )
        probe = self.log_likelihood_fn(theta_probe, x_obs_batch[:1])
        device = probe.device
        if x_obs_batch.device != device:
            x_obs_batch = x_obs_batch.to(device)
        B = x_obs_batch.shape[0]
        lo, hi = self.theta_range

        def _compute_ll_grid_and_max():
            if self.d_theta == 1:
                theta_grid = torch.linspace(
                    lo, hi, n_grid, device=device, dtype=x_obs_batch.dtype,
                ).view(-1, 1)
            else:
                grid_np = np.random.default_rng(0).uniform(lo, hi, size=(n_grid, self.d_theta))
                theta_grid = torch.from_numpy(grid_np).to(
                    dtype=x_obs_batch.dtype, device=device,
                )
            theta_grid_exp = theta_grid.unsqueeze(0).expand(B, -1, -1).reshape(-1, self.d_theta)
            x_obs_exp = (
                x_obs_batch.unsqueeze(1).expand(-1, n_grid, -1).reshape(-1, x_obs_batch.shape[-1])
            )
            ll_g = self.log_likelihood_fn(theta_grid_exp, x_obs_exp).view(B, n_grid)
            return theta_grid, ll_g, ll_g.max(dim=1).values

        cache_key = ("ll_grid", id(x_obs_batch), int(n_grid))
        theta_grid, ll_grid, ll_max = self._ll_grid_cache.get_or_compute(
            cache_key, _compute_ll_grid_and_max,
        )
        # ll at the candidate θ_0, batched over X_obs:
        theta_0_t = _theta_to_row_tensor(
            theta_0_value, x_obs_batch.dtype, device, self.d_theta,
        ).expand(B, -1)
        ll_at_0 = self.log_likelihood_fn(theta_0_t, x_obs_batch)
        if ll_at_0.ndim > 1:
            ll_at_0 = ll_at_0.squeeze(-1)
        return 2.0 * (ll_max - ll_at_0) <= thresh

    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        if self.d_theta == 1:
            return self._confidence_set_1d(x_obs, alpha)
        return self._confidence_set_d_gt_1(x_obs, alpha)

    def _confidence_set_1d(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        thresh = float(chi2.ppf(alpha, df=1))

        lo, hi = self.theta_range
        grid = torch.linspace(lo, hi, 200).view(-1, 1).to(x_obs.device)
        lls = torch.stack([self.log_likelihood_fn(g.view(1, 1), x_obs).squeeze() for g in grid])
        ll_max = lls.max().item()

        def f(theta_val: float) -> float:
            theta = torch.tensor([[theta_val]], dtype=x_obs.dtype, device=x_obs.device)
            ll = self.log_likelihood_fn(theta, x_obs).item()
            return 2 * (ll_max - ll) - thresh

        grid_f = [2 * (ll_max - ll.item()) - thresh for ll in lls]
        signs = [g <= 0 for g in grid_f]
        try:
            i_first = signs.index(True)
            i_last = len(signs) - 1 - signs[::-1].index(True)
        except ValueError:
            return ConfidenceSet(
                contains=lambda th: False, boundary_repr=torch.tensor([0.0, 0.0]), alpha=alpha
            )
        gs = grid.flatten().tolist()
        left_lo, left_hi = gs[max(i_first - 1, 0)], gs[i_first]
        right_lo, right_hi = gs[i_last], gs[min(i_last + 1, len(gs) - 1)]
        left = bisect_1d(f, left_lo, left_hi, tol=1e-4) if i_first > 0 else gs[0]
        right = bisect_1d(f, right_lo, right_hi, tol=1e-4) if i_last < len(gs) - 1 else gs[-1]

        def contains(theta_val: float) -> bool:
            return left <= theta_val <= right

        return ConfidenceSet(
            contains=contains, boundary_repr=torch.tensor([left, right]), alpha=alpha
        )

    def _confidence_set_d_gt_1(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        thresh = float(chi2.ppf(alpha, df=self.d_theta))
        lo, hi = self.theta_range
        dtype = x_obs.dtype
        device = x_obs.device
        # Precompute ll_max(X) via the same coarse θ-grid the ray helper uses
        # (cheap, and stays consistent with what _ray_sample_set_boundary will
        # refine via L-BFGS).
        n_grid = 25  # coarse but enough for ll_max approx in d=2-3
        axes = [torch.linspace(lo, hi, n_grid, dtype=dtype, device=device)
                for _ in range(self.d_theta)]
        mesh = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).view(-1, self.d_theta)
        with torch.no_grad():
            x_batch_for_ll_max = x_obs.expand(mesh.shape[0], -1)
            ll_grid = self.log_likelihood_fn(mesh, x_batch_for_ll_max)
            ll_max = float(ll_grid.max().item())

        def inside_fn(theta, x):
            ll = self.log_likelihood_fn(theta, x)
            return 2.0 * (ll_max - ll) - thresh

        contains_fn, boundary = _ray_sample_set_boundary(
            x_obs, inside_fn, self.theta_range, self.d_theta, n_rays=200,
        )
        return ConfidenceSet(contains=contains_fn, boundary_repr=boundary, alpha=alpha)

    def confidence_set_batch(
        self, x_obs_batch: torch.Tensor, alpha: float, n_grid: int = 200,
    ) -> tuple:
        """Vectorized 1D Wilks-LR confidence interval boundaries across a batch
        of X_obs. Returns (left, right), each shape (B,). 1D only in v1.
        """
        assert self.d_theta == 1, "confidence_set_batch is 1D only in v1"
        thresh = float(chi2.ppf(alpha, df=1))
        # Probe device.
        probe = self.log_likelihood_fn(
            torch.zeros((1, 1), dtype=x_obs_batch.dtype, device=x_obs_batch.device),
            x_obs_batch[:1],
        )
        device = probe.device
        if x_obs_batch.device != device:
            x_obs_batch = x_obs_batch.to(device)
        B = x_obs_batch.shape[0]
        lo, hi = self.theta_range

        # 1) Coarse-grid ll evaluation, B × n_grid — α-invariant, so cache it.
        def _compute_ll_grid_and_max():
            theta_grid_ = torch.linspace(
                lo, hi, n_grid, device=device, dtype=x_obs_batch.dtype,
            ).view(-1, 1)
            theta_grid_exp = theta_grid_.unsqueeze(0).expand(B, -1, -1).reshape(-1, 1)
            x_obs_exp = x_obs_batch.unsqueeze(1).expand(-1, n_grid, -1).reshape(
                -1, x_obs_batch.shape[-1]
            )
            ll_g = self.log_likelihood_fn(theta_grid_exp, x_obs_exp).view(B, n_grid)
            return theta_grid_, ll_g, ll_g.max(dim=1).values

        cache_key = ("ll_grid", id(x_obs_batch), int(n_grid))
        theta_grid, ll_grid, ll_max_per_row = self._ll_grid_cache.get_or_compute(
            cache_key, _compute_ll_grid_and_max,
        )
        center_idx = ll_grid.argmax(dim=1)  # (B,)
        center_theta = theta_grid.squeeze(-1)[center_idx]  # (B,)

        # 2) f(θ; X) = 2 (ll_max - ll(θ)) - thresh; set = {f ≤ 0}.
        def f_batch(theta_vals: torch.Tensor) -> torch.Tensor:
            theta_b = theta_vals.unsqueeze(-1)  # (B, 1)
            ll = self.log_likelihood_fn(theta_b, x_obs_batch).squeeze(-1)
            return 2.0 * (ll_max_per_row - ll) - thresh

        # 3) Bisect LEFT root on [lo, center]: f(lo) > 0 (typically), f(center) ≤ 0.
        a = torch.full((B,), lo, dtype=x_obs_batch.dtype, device=device)
        b = center_theta.clone()
        left = _vectorized_bisect_1d(f_batch, a, b, f_sign_at_a=+1)
        # 4) Bisect RIGHT root on [center, hi]: f(center) ≤ 0, f(hi) > 0.
        a = center_theta.clone()
        b = torch.full((B,), hi, dtype=x_obs_batch.dtype, device=device)
        right = _vectorized_bisect_1d(f_batch, a, b, f_sign_at_a=-1)
        return left, right


class RatioBasedProcedure:
    """NRE: ratio thresholding — same shape as Likelihood but using log-ratio."""

    def __init__(self, log_ratio_fn: Callable, d_theta: int, theta_range: tuple = (-20.0, 20.0)):
        self.log_ratio_fn = log_ratio_fn
        self.d_theta = d_theta
        self.theta_range = theta_range
        self._likelihood_wrapper = None

    def _wrapper(self) -> "LikelihoodBasedProcedure":
        if self._likelihood_wrapper is None:
            self._likelihood_wrapper = LikelihoodBasedProcedure(
                self.log_ratio_fn, self.d_theta, self.theta_range,
            )
        return self._likelihood_wrapper

    def log_ratio(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.log_ratio_fn(theta, x)

    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        return self._wrapper().confidence_set(x_obs, alpha)

    def contains_batch(
        self, theta_0_value, x_obs_batch: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        return self._wrapper().contains_batch(theta_0_value, x_obs_batch, alpha)

    def confidence_set_batch(
        self, x_obs_batch: torch.Tensor, alpha: float
    ) -> tuple:
        return self._wrapper().confidence_set_batch(x_obs_batch, alpha)
