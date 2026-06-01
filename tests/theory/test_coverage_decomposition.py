"""Numerical validators for docs/theory/2026-05-31-conditional-coverage-decomposition.md.

These tests pin the *inferential logic* of CD-SBI to runnable assertions. Each
test names the theorem/lemma it validates. The closed-form pivot r* of
NormalUnknownMeanVar (the first scale/nuisance target) is the workhorse: it lets
us check the population-optimum (C2 exact) statements directly, with no training.

Tests that require trained runs or new toy simulators are scaffolded with
pytest.skip + a precise spec, so "scaffold as we go" is literal: the next session
fills them in against the named artifacts.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.stats import chi2, kstest, norm

from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar


# θ₀ probes across the prior box, including the corners where the average-vs-sup
# gap (Prop 6) would bite a *trained* model. The oracle r* must cover at ALL of
# them — that is the content of "everywhere" for the population optimum.
_THETA0_PROBES = [
    (math.log(1.0), 0.0),     # center-ish
    (math.log(0.3), -5.0),    # box corner: smallest σ, most negative μ
    (math.log(3.0), 5.0),     # box corner: largest σ, most positive μ
    (math.log(0.5), 3.0),     # mixed
]
_ALPHAS = (0.5, 0.8, 0.9, 0.95)
_N_SAMPLES = 40_000


def _pivot_at_theta0(sim, theta0, n, seed):
    """Draw X ~ p(·|θ₀) and return r*(θ₀; X), shape (n, d)."""
    rng = np.random.default_rng(seed)
    x = sim.sample_x_given_theta(np.asarray(theta0), n, rng)            # (n, n_iid)
    theta = torch.tensor(theta0, dtype=torch.float32).expand(n, -1)     # (n, d)
    return sim.r_star(theta, x)


# ---------------------------------------------------------------------------
# Theorem 1 — C1 + C2  ⟹  exact conditional coverage, for EVERY θ₀.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theta0", _THETA0_PROBES)
def test_theorem1_oracle_exact_coverage_everywhere(theta0):
    """At the population optimum (oracle r*, so C2 exact; r* is a sign-correct
    diffeomorphism in (log s², X̄), so C1 holds), the pushforward is N(0,I) and
    C_α covers at exactly α — at every θ₀ including box corners."""
    sim = NormalUnknownMeanVar()
    r = _pivot_at_theta0(sim, theta0, _N_SAMPLES, seed=0).numpy()

    # (a) per-coordinate standard normal (KS) — the pushforward claim.
    for k in range(r.shape[1]):
        ks_p = kstest(r[:, k], "norm").pvalue
        assert ks_p > 1e-3, f"coord {k} not N(0,1) at θ₀={theta0}: KS p={ks_p:.2e}"

    # (b) coverage of C_α = {‖r‖² ≤ χ²_{d,α}} equals α (the inferential payload).
    sq = (r ** 2).sum(axis=1)                       # ‖r(θ₀;X)‖² ~ χ²_d
    d = r.shape[1]
    se = 1.0 / math.sqrt(_N_SAMPLES)                # binomial SE of a coverage frac
    for a in _ALPHAS:
        emp = float((sq <= chi2.ppf(a, df=d)).mean())
        assert abs(emp - a) < 5 * se, f"coverage {emp:.4f} ≠ {a} at θ₀={theta0}"


# ---------------------------------------------------------------------------
# Lemma 3 / Cor 4 — calibration is sign-blind; the (↓θ, ↑data) branch is ALSO
# exactly calibrated. Hard-coding (↑θ, ↑data) excludes BOTH valid branches.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theta0", _THETA0_PROBES)
def test_reflection_branch_also_calibrates(theta0):
    """r* uses the σ-branch Φ⁻¹(1−F): (↑ in θ_σ, ↓ in data). The reflection
    r_alt = [−r*_σ, r*_μ] is the OTHER scale branch (↓ in θ_σ, ↑ in data). It
    Gaussianizes the same conditional, so it has identical exact coverage — the
    point of Cor 4: '↑data' is satisfiable, just not jointly with '↑θ'."""
    sim = NormalUnknownMeanVar()
    r = _pivot_at_theta0(sim, theta0, _N_SAMPLES, seed=1)
    r_alt = torch.stack([-r[:, 0], r[:, 1]], dim=-1)    # flip ONLY the scale coord

    # ‖·‖² is invariant under a coordinate sign flip ⟹ identical coverage.
    sq, sq_alt = (r ** 2).sum(-1).numpy(), (r_alt ** 2).sum(-1).numpy()
    d = 2
    for a in _ALPHAS:
        thr = chi2.ppf(a, df=d)
        cov, cov_alt = float((sq <= thr).mean()), float((sq_alt <= thr).mean())
        assert abs(cov - cov_alt) < 1e-12, "reflected branch must cover identically"
    # and the flipped coord is still standard normal (sign-blind calibration).
    assert kstest(r_alt[:, 0].numpy(), "norm").pvalue > 1e-3


# ---------------------------------------------------------------------------
# Theorem 5 — oracle (zero-gap) end: at r*, L ≈ H̄ and coverage error ≈ 0, and
# crucially L ≥ H̄ − noise (no folding ⟺ C1 holds ⟺ the certificate is honest).
# ---------------------------------------------------------------------------
def test_theorem5_oracle_zero_gap_and_honest_floor():
    """The NF-MLE loss evaluated at the oracle pivot r* sits AT the entropy floor
    (gap ≈ 0), never below it. L < H̄ would signal a folding/C1 violation; the
    oracle is injective by construction, so the certificate sqrt((L−H̄)/2) is ~0,
    matching the ~0 measured coverage error from Theorem 1."""
    sim = NormalUnknownMeanVar()
    H_bar = sim.entropy_lower_bound()                      # E[NF-MLE loss at r*]

    # Independent MC estimate of E[loss at r*] using the analytic feature-Jacobian
    # of r* (same construction the floor uses, recomputed via the loss object as a
    # cross-check that L − H̄ ≈ 0 with no folding).
    rng = np.random.default_rng(7)
    theta, x = sim.sample(_N_SAMPLES, rng)
    r = sim.r_star(theta, x)                               # (n, 2)
    xbar, s2 = sim._suff_stats(x)
    sigma = torch.exp(theta[:, 0:1])
    n = sim.n_iid
    w = ((n - 1) * s2 / sigma ** 2)
    # |∂r_σ/∂ log s²| = w f_χ²(w)/φ(r_σ);  |∂r_μ/∂X̄| = √n/σ  (lower-triangular)
    f_chi2 = torch.from_numpy(chi2.pdf(w.numpy(), df=n - 1)).float()
    phi_rs = torch.from_numpy(norm.pdf(r[:, 0:1].numpy())).float().clamp_min(1e-30)
    dr_sigma = (w * f_chi2 / phi_rs)
    dr_mu = math.sqrt(n) / sigma
    log_det = (torch.log(dr_sigma.clamp_min(1e-30)) + torch.log(dr_mu)).squeeze(-1)

    L = float(NFMLELoss()(r, log_det))
    gap = L - H_bar
    # gap should be ~0 (two MC estimates of the same expectation) and not strongly
    # negative (a strongly-negative gap is the folding signature).
    assert gap > -0.05, f"L below floor by {-gap:.3f} — folding/C1 violation?"
    assert abs(gap) < 0.10, f"oracle gap {gap:.3f} not ≈ 0 (L={L:.3f}, H̄={H_bar:.3f})"
    # certificate is tiny ⟹ consistent with the ~0 coverage error of Theorem 1.
    assert math.sqrt(max(gap, 0.0) / 2) < 0.25


# ===========================================================================
# Scaffolds — specified precisely, filled in next as the program advances.
# ===========================================================================
@pytest.mark.skip(reason="SCAFFOLD (S3/Thm5 trained end): load a trained sweep and "
                  "assert measured sup_α|cov−α| ≤ sqrt((L−H̄)/2) per (method,budget,seed). "
                  "Source: outputs/8_*_baseline_sweep/* via "
                  "cdsbi.analysis.figures.data_io.figure_data.load_sweep; floor from "
                  "simulator.entropy_lower_bound(). Validates the certificate end-to-end.")
def test_theorem5_bound_holds_for_trained_runs():
    ...


# ---------------------------------------------------------------------------
# Proposition 2 — validity ⟂ sufficiency. A NON-sufficient summary still gives
# EXACT coverage; sufficiency only controls set volume.
#
# Construction: the "first-m-obs" pivot. Using only X_1..X_m (m < n_iid) of the
# n_iid observations, the stats (s²_m, X̄_m) are STILL a Basu-independent
# (χ²_{m-1}, N) pair at the truth, so r^(m)(θ₀;X) ~ N(0,I₂) exactly — exact
# coverage — even though the summary discards X_{m+1..n} (non-sufficient).
# ---------------------------------------------------------------------------
def _subsample_pivot(theta0, x, m):
    """Exact pivot built from only the first m of n_iid obs. Same closed form as
    r* with n→m. theta0 = (log σ, μ)."""
    xm = x[:, :m]
    xbar = xm.mean(dim=-1, keepdim=True)
    s2 = xm.var(dim=-1, unbiased=True, keepdim=True)
    log_sigma, mu = float(theta0[0]), float(theta0[1])
    sigma = math.exp(log_sigma)
    w = ((m - 1) * s2 / sigma ** 2).numpy()
    u = np.clip(chi2.cdf(w, df=m - 1), 1e-12, 1 - 1e-12)
    r_sigma = torch.from_numpy(norm.ppf(1.0 - u)).float()
    r_mu = math.sqrt(m) * (mu - xbar) / sigma
    return torch.cat([r_sigma, r_mu], dim=-1)


@pytest.mark.parametrize("theta0", _THETA0_PROBES)
@pytest.mark.parametrize("m", [5, 10])
def test_prop2_nonsufficient_summary_still_covers(theta0, m):
    """m=10 is the sufficient summary; m=5 is non-sufficient (drops half the
    data). BOTH cover at exactly α — validity does not depend on sufficiency."""
    sim = NormalUnknownMeanVar()
    rng = np.random.default_rng(11)
    x = sim.sample_x_given_theta(np.asarray(theta0), _N_SAMPLES, rng)
    r = _subsample_pivot(theta0, x, m).numpy()
    sq = (r ** 2).sum(axis=1)
    se = 1.0 / math.sqrt(_N_SAMPLES)
    for a in _ALPHAS:
        emp = float((sq <= chi2.ppf(a, df=2)).mean())
        assert abs(emp - a) < 5 * se, f"m={m} θ₀={theta0}: coverage {emp:.4f} ≠ {a}"


def _theta_jacobian_logdet(theta0, x, m, eps=1e-3):
    """E[ log|det ∂r/∂θ| ] by central differences — the local log-volume
    COMPRESSION of the pivot. Larger ⟹ tighter confidence sets (set volume
    ≈ χ²-ball / |det ∂r/∂θ|). This is the efficiency surrogate."""
    cols = []
    for j in range(2):
        tp, tm = list(theta0), list(theta0)
        tp[j] += eps
        tm[j] -= eps
        d = (_subsample_pivot(tp, x, m) - _subsample_pivot(tm, x, m)) / (2 * eps)
        cols.append(d)                                   # ∂r/∂θ_j, shape (n,2)
    J = torch.stack(cols, dim=-1)                        # (n,2,2)
    logdet = torch.log(torch.linalg.det(J).abs().clamp_min(1e-30))
    return float(logdet.mean())


def test_prop2_sufficiency_buys_efficiency():
    """The sufficient summary (m=10) compresses θ MORE than the non-sufficient
    (m=5) — larger E[log|det ∂r/∂θ|] ⟹ smaller sets. Validity is identical
    (previous test); only volume differs. Expected gain ≳ ½·log(10/5)=0.347 from
    the μ-coordinate's √m alone."""
    sim = NormalUnknownMeanVar()
    rng = np.random.default_rng(12)
    theta0 = (math.log(1.0), 0.0)
    x = sim.sample_x_given_theta(np.asarray(theta0), _N_SAMPLES, rng)
    ld10 = _theta_jacobian_logdet(theta0, x, m=10)
    ld5 = _theta_jacobian_logdet(theta0, x, m=5)
    assert ld10 > ld5 + 0.30, (
        f"sufficient summary should be tighter: log|det∂r/∂θ| m=10 {ld10:.3f} "
        f"vs m=5 {ld5:.3f}"
    )


@pytest.mark.intensive
def test_s3_validity_phi_free_trained():
    """S3 — the genuinely *learned*-on-a-non-sufficient-summary claim. Train a
    SingleIndexMonotoneFlow by NF-MLE on the LOSSY summary (log s², X̄) computed
    from only the first m=5 obs, fresh-batch (so q→p). Theorem 1 then predicts
    exact coverage despite the summary being non-sufficient. Validity is φ-free."""
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow

    sim = NormalUnknownMeanVar()
    m = 5

    def feats_lossy(x):
        xm = x[:, :m]
        xbar = xm.mean(dim=-1, keepdim=True)
        s2 = xm.var(dim=-1, unbiased=True, keepdim=True)
        return torch.cat([torch.log(s2.clamp_min(1e-12)), xbar], dim=-1)

    torch.manual_seed(0)
    flow = SingleIndexMonotoneFlow(
        d=2, theta_signs=sim.theta_signs, feat_signs=sim.feat_signs, hidden=32,
    )
    loss_fn = NFMLELoss()
    opt = torch.optim.Adam(flow.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6000)
    rng = np.random.default_rng(0)
    flow.train()
    for step in range(6000):
        theta, x = sim.sample(512, rng)
        r, log_det = flow(theta, feats_lossy(x))
        loss = loss_fn(r, log_det)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    flow.eval()
    for theta0 in _THETA0_PROBES:
        xv = sim.sample_x_given_theta(np.asarray(theta0), 20_000, rng)
        th = torch.tensor(theta0, dtype=torch.float32).expand(20_000, -1)
        with torch.no_grad():
            r, _ = flow(th, feats_lossy(xv))
        sq = (r ** 2).sum(-1).numpy()
        for a in (0.8, 0.9, 0.95):
            emp = float((sq <= chi2.ppf(a, df=2)).mean())
            assert abs(emp - a) < 0.06, (
                f"learned lossy-summary flow miscovers at θ₀={theta0}, "
                f"α={a}: {emp:.4f}"
            )


# ---------------------------------------------------------------------------
# Theorem S2 — a sup-controlling (minimax) objective yields uniform coverage
# where the ρ-average objective (plain NF-MLE) leaves a tail gap.
#
# Deterministic closed-form witness: finite θ-grid, T|θ ~ N(0, v_θ), and a
# MIS-SPECIFIED single-scale pivot r(θ;T) = T/s (one s for all θ — a stand-in for
# a capacity-limited flow that cannot calibrate every θ at once). Pushforward is
# N(0, v_θ/s²); calibration at θ ⟺ s² = v_θ. With one s, you trade θ's off — the
# exact average-vs-sup tension of Prop 6 / μ₂, here solvable in closed form.
# ---------------------------------------------------------------------------
def _kl_gauss(v, s2):
    """KL( N(0,v) ‖ N(0,s²) ) = ½(v/s² − 1 − log(v/s²))."""
    rr = v / s2
    return 0.5 * (rr - 1.0 - np.log(rr))


def _sup_coverage_error(s2, v, alphas=(0.8, 0.9, 0.95)):
    """sup over θ and α of |coverage − α| for the χ²₁ set with scale s²."""
    e = 0.0
    for a in alphas:
        cov = chi2.cdf((s2 / v) * chi2.ppf(a, df=1), df=1)
        e = max(e, float(np.max(np.abs(cov - a))))
    return e


def test_s2_minimax_controls_sup_coverage():
    """The ρ-average optimum is s²=E_ρ[v] (plain NF-MLE: minimize E_ρ KL). With ρ
    underweighting the high-variance tail, it badly under-covers there. The
    minimax optimum (minimize sup_θ KL) balances the extremes and roughly halves
    the sup-coverage error — converting 'coverage on average' to 'coverage
    everywhere'. Both obey the pointwise Pinsker certificate sup-cov ≤ √(supKL/2)."""
    from scipy.optimize import minimize_scalar

    v = np.array([1.0, 1.0, 1.0, 1.0, 4.0])          # one high-variance tail θ
    rho = np.array([0.24, 0.24, 0.24, 0.24, 0.04])   # ρ underweights that tail

    s2_avg = float(rho @ v)                           # argmin_s² E_ρ[KL]  (closed form)
    s2_mm = float(minimize_scalar(                    # argmin_s² sup_θ KL
        lambda s2: float(np.max(_kl_gauss(v, s2))),
        bounds=(0.3, 4.0), method="bounded").x)

    sup_avg = _sup_coverage_error(s2_avg, v)
    sup_mm = _sup_coverage_error(s2_mm, v)

    # (1) minimax strictly improves the WORST-θ coverage error.
    assert sup_mm < sup_avg - 0.05, (
        f"minimax sup-cov {sup_mm:.3f} should beat average {sup_avg:.3f}"
    )
    # (2) pointwise Pinsker certificate holds for BOTH objectives (Theorem S2).
    for s2 in (s2_avg, s2_mm):
        bound = math.sqrt(float(np.max(_kl_gauss(v, s2))) / 2)
        assert _sup_coverage_error(s2, v) <= bound + 1e-9, "Pinsker bound violated"
    # (3) the average objective is genuinely deficient on the sup (else no problem
    #     to solve) — guards the test against a degenerate v/ρ choice.
    assert sup_avg > 0.15


# ---------------------------------------------------------------------------
# Theorem 7(iii) — the collapse. A θ-INDEPENDENT χ²_d statistic gives EXACT
# coverage at every θ₀ with whole-space-or-empty sets: coverage is blind to
# summary collapse. ILLUSTRATION of the proof (which is the actual result), here
# realized with an external χ²_d draw (the in-framework version de-randomizes it
# via an ancillary statistic — Thm 7(iii)). Coverage objective J=0, zero power.
# ---------------------------------------------------------------------------
def test_thm7_collapse_exact_coverage_zero_power():
    d = 2
    rng = np.random.default_rng(3)
    # V ~ χ²_d, drawn independently of θ — stands in for ‖g(ancillary)‖².
    V = rng.chisquare(df=d, size=_N_SAMPLES)
    se = 1.0 / math.sqrt(_N_SAMPLES)
    # (1) exact coverage at EVERY θ₀ (the statistic ignores θ₀ entirely).
    for theta0 in _THETA0_PROBES:
        for a in _ALPHAS:
            emp = float((V <= chi2.ppf(a, df=d)).mean())
            assert abs(emp - a) < 5 * se, f"collapse should still cover: {emp:.4f} vs {a}"
    # (2) zero power: the acceptance indicator is identical for ALL θ₀ (the set is
    #     {∅,Θ}), so it cannot exclude any false θ.
    accept = {tuple(t): (V <= chi2.ppf(0.9, df=d)) for t in _THETA0_PROBES}
    ref = accept[tuple(_THETA0_PROBES[0])]
    for t in _THETA0_PROBES[1:]:
        assert np.array_equal(accept[tuple(t)], ref), "set must be θ-independent (∅ or Θ)"


# ---------------------------------------------------------------------------
# Theorems 8 & 9 (#2) — closed-form minimax–average separation for the
# single-scale family, and the Sion least-favorable-Bayes duality. These CHECK
# the closed forms in the note against direct numerical optimization; the proofs
# (note §9½) are the result.
# ---------------------------------------------------------------------------
def _R_gauss(v, u):
    """KL( N(0,v) ‖ N(0,u) ) = ½(v/u − 1 + log(u/v))."""
    return 0.5 * (v / u - 1.0 + np.log(u / v))


def test_thm8_minimax_average_closed_form():
    from scipy.optimize import minimize_scalar
    a, b = 1.0, 4.0

    u_star = (b - a) / math.log(b / a)            # logarithmic mean (closed form)
    eps_star = _R_gauss(a, u_star)

    # (2) minimax: R(a;u*)=R(b;u*) (balanced) and matches numeric inf_u max.
    assert abs(_R_gauss(a, u_star) - _R_gauss(b, u_star)) < 1e-12
    num = minimize_scalar(lambda u: max(_R_gauss(a, u), _R_gauss(b, u)),
                          bounds=(a, b), method="bounded")
    assert abs(num.x - u_star) < 1e-4 and abs(num.fun - eps_star) < 1e-6

    # (1) average optimum = arithmetic mean E_ρ[v]; (3) strict separation + exact gap.
    v = np.array([1., 1., 1., 1., 4.]); rho = np.array([.24, .24, .24, .24, .04])
    u_avg = float(rho @ v)
    assert abs(u_avg - 1.12) < 1e-9
    sup_avg = max(_R_gauss(a, u_avg), _R_gauss(b, u_avg))
    assert sup_avg > eps_star + 0.3                       # strict, large gap
    # exact one-sided gap formula (u_avg < u* here ⟹ governed by the b-endpoint).
    assert u_avg < u_star
    assert abs(sup_avg - _R_gauss(b, u_avg)) < 1e-12


def test_thm9_sion_least_favorable():
    a, b = 1.0, 4.0
    u_star = (b - a) / math.log(b / a)
    eps_star = _R_gauss(a, u_star)

    # least-favorable 2-point prior on {a,b}: weight w on b with E_πLF[v] = u*.
    w_lf = (u_star - a) / (b - a)
    assert abs((1 - w_lf) * a + w_lf * b - u_star) < 1e-12

    # Sion duality: sup_π inf_u E_π[R] = ε*. Bayes-opt u under π(w) is E_π[v].
    def bayes_risk(w):
        u = (1 - w) * a + w * b
        return (1 - w) * _R_gauss(a, u) + w * _R_gauss(b, u)
    ws = np.linspace(1e-3, 1 - 1e-3, 20001)
    i = int(np.argmax([bayes_risk(w) for w in ws]))
    assert abs(ws[i] - w_lf) < 2e-3
    assert abs(bayes_risk(ws[i]) - eps_star) < 1e-5        # = ε* by Sion


# ---------------------------------------------------------------------------
# Theorems 10 & 11 (§11, S4 foundation) — efficiency = Fisher-information
# preservation. For N(μ,σ²), n iid, coords θ=(log σ, μ): the analytic Fisher
# information is I = diag(n/σ², 2n), det I = 2n²/σ². A first-m-obs summary has
# I_m = (m/n)·I (iid additivity) — strictly less (Thm 10), and Thm 11 predicts
# the set-volume gap. These CHECK the closed forms; §11 proofs are the result.
# ---------------------------------------------------------------------------
def _fisher_mu_sigma(n, sigma=1.0):
    """Analytic Fisher info of N(μ,σ²), n iid, in coords (log σ, μ)."""
    return np.diag([2.0 * n, n / sigma ** 2])         # (log σ, μ) order


def test_thm10_fisher_data_processing():
    n, m = 10, 5
    I_full = _fisher_mu_sigma(n)
    I_sub = _fisher_mu_sigma(m)                        # = (m/n) I_full
    assert np.allclose(I_sub, (m / n) * I_full)
    # Loewner I_full ⪰ I_sub, STRICT (non-sufficient summary ⟹ strict, Thm 10).
    assert np.all(np.linalg.eigvalsh(I_full - I_sub) > 0)
    # det ratio ⟹ volume ratio (n/m)^{d/2} (Thm 11).
    assert abs(np.linalg.det(I_full) / np.linalg.det(I_sub) - (n / m) ** 2) < 1e-9


def _diag_jac_logs(theta0, x, m, eps=1e-3):
    """Per-coordinate E[log|∂r_k/∂θ_k|] for the sub-sample pivot, k=(σ,μ)."""
    out = []
    for k in range(2):
        tp, tm = list(theta0), list(theta0)
        tp[k] += eps
        tm[k] -= eps
        d = (_subsample_pivot(tp, x, m)[:, k] - _subsample_pivot(tm, x, m)[:, k]) / (2 * eps)
        out.append(float(torch.log(d.abs().clamp_min(1e-30)).mean()))
    return out[0], out[1]                              # (σ-coord, μ-coord)


def test_thm11_fisher_exact_for_location_asymptotic_for_scale():
    """Thm 11 (log|∂r_k/∂θ_k| = ½ log I_kk) is EXACT for the location coord and
    ASYMPTOTIC for the scale coord (the diagnostic that turned the naive 'gap=log2'
    claim into the honest statement). For N(μ,σ²), m obs, σ=1: I_μμ=m ⟹ ½log m;
    I_σσ=2m ⟹ ½log(2m)."""
    sim = NormalUnknownMeanVar()
    rng = np.random.default_rng(5)
    theta0 = (math.log(1.0), 0.0)
    x = sim.sample_x_given_theta(np.asarray(theta0), 120_000, rng)

    defs = {}
    for m in (5, 10):
        ls_sigma, ls_mu = _diag_jac_logs(theta0, x, m)
        # (1) LOCATION: exact at every m (∂r_μ/∂μ = √m/σ, deterministic).
        assert abs(ls_mu - 0.5 * math.log(m)) < 5e-3, f"μ not exact at m={m}"
        # scale Fisher deficit ½log(2m) − measured (>0, shrinking).
        defs[m] = 0.5 * math.log(2 * m) - ls_sigma
    # (2) SCALE: positive finite-n deficit that shrinks ~O(1/n) (χ²→Gaussian).
    assert defs[5] > 0 and defs[10] > 0
    assert defs[10] < 0.65 * defs[5], f"scale deficit not shrinking: {defs}"


# ---------------------------------------------------------------------------
# §13.3 — KL (validity) does NOT control the score (efficiency). A high-frequency
# density perturbation keeps KL = O(ε²) for all ω while the score error grows ~εω.
# ⟹ S4's efficiency term needs explicit score control, not just NF-MLE. (Proven
# negative result; this is its witness.)
# ---------------------------------------------------------------------------
def test_thm13_kl_does_not_control_score():
    xs = np.linspace(-9, 9, 600_000); dx = xs[1] - xs[0]
    p = np.exp(-xs ** 2 / 2) / math.sqrt(2 * math.pi); eps = 0.1

    def kl_scorediv(omega):
        wig = 1 + eps * np.sin(omega * xs)
        q = p * wig / (np.sum(p * wig) * dx)
        kl = float(np.sum(p * np.log(np.clip(p / q, 1e-300, None))) * dx)
        dlogwig = eps * omega * np.cos(omega * xs) / wig     # score_q − score_p
        return kl, float(np.sum(p * dlogwig ** 2) * dx)

    kl_lo, sd_lo = kl_scorediv(2.0)
    kl_hi, sd_hi = kl_scorediv(20.0)
    # KL essentially unchanged across a 10× frequency increase ...
    assert abs(kl_lo - kl_hi) < 1e-3
    # ... while the score divergence grows ~ω² (≈100×): unbounded by KL.
    assert sd_hi > 50 * sd_lo


# ---------------------------------------------------------------------------
# §14.2 — Thm 11's set-volume formula, exact Gaussian case. For T~N(θ,Σ_T) the
# calibrated pivot's set is the ellipse {(θ−T)ᵀΣ_T⁻¹(θ−T) ≤ χ²}, whose area is
# V_2·χ²·det(Σ_T)^{1/2} = V_2·χ²·det I_T^{−1/2}. Confirms volume ↔ Fisher info.
# ---------------------------------------------------------------------------
def test_thm11_gaussian_volume_exact():
    Sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    Sinv = np.linalg.inv(Sigma)
    q = chi2.ppf(0.9, df=2)
    predicted = math.pi * q * math.sqrt(np.linalg.det(Sigma))   # V_2 χ² det I_T^{-1/2}

    g = np.linspace(-12, 12, 2401); dt = g[1] - g[0]
    TH = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    quad = np.einsum("ni,ij,nj->n", TH, Sinv, TH)               # ‖r(θ;T=0)‖²
    grid_area = float((quad <= q).sum()) * dt * dt
    assert abs(grid_area - predicted) < 0.05, f"{grid_area:.4f} vs {predicted:.4f}"
    # det I_T^{-1/2} = det(Σ_T)^{1/2} (I_T = Σ_T^{-1}) — the efficiency link.
    assert abs(math.sqrt(np.linalg.det(Sigma)) - math.sqrt(1.0 / np.linalg.det(Sinv))) < 1e-9


# ---------------------------------------------------------------------------
# Theorem 15 (§15) — multimodal is NOT impossible for CD-SBI. The sign-
# unidentifiable model X̄|θ~N(θ²,σ²/n) (bimodal posterior) is handled exactly by a
# 1-dim SUFFICIENT summary X̄ + a NON-monotone pivot: valid + efficient +
# correctly DISCONNECTED. R1 (monotone) is the only thing that would forbid it.
# ---------------------------------------------------------------------------
def test_thm15_multimodal_handled_by_nonmonotone_pivot():
    from scipy.stats import kstest
    n, sig = 20, 1.0
    rng = np.random.default_rng(0)
    se = 1.0 / math.sqrt(_N_SAMPLES)
    for theta0 in (1.5, 0.8, 2.0):
        xbar = rng.normal(theta0 ** 2, sig / math.sqrt(n), size=_N_SAMPLES)
        r = math.sqrt(n) * (xbar - theta0 ** 2) / sig        # non-monotone in θ
        # (1) valid: pivot N(0,1) at truth ⟹ coverage = α
        assert kstest(r, "norm").pvalue > 1e-3
        for a in (0.8, 0.9, 0.95):
            emp = float((r ** 2 <= chi2.ppf(a, df=1)).mean())
            assert abs(emp - a) < 5 * se
        # (2) disconnected: C_α = {θ: θ² ∈ X̄ ± z σ/√n} is two disjoint intervals
        #     whenever the lower band edge is positive — the correct bimodal answer.
        z = math.sqrt(chi2.ppf(0.9, df=1))
        lo = xbar - z * sig / math.sqrt(n)
        assert (lo > 0).mean() > 0.8


# ---------------------------------------------------------------------------
# Theorem 16 (§16) — the mode-resolving summary is posterior-MOMENT regression.
# In the bimodal sign model the 1st moment (posterior mean) target is useless, but
# the 2nd moment recovers the sufficient statistic. Moment-order = the modality knob.
# ---------------------------------------------------------------------------
def test_thm16_posterior_moment_recovers_sufficiency():
    n, sig = 20, 1.0
    rng = np.random.default_rng(0)
    N = 300_000
    theta = rng.uniform(-3, 3, size=N)
    xbar = rng.normal(theta ** 2, sig / math.sqrt(n), size=N)
    # regression target = a posterior moment; corr with the sufficient stat X̄ measures
    # how much of the sufficiency that moment recovers.
    c1 = abs(float(np.corrcoef(theta, xbar)[0, 1]))       # 1st moment (posterior MEAN)
    c2 = float(np.corrcoef(theta ** 2, xbar)[0, 1])       # 2nd moment
    assert c1 < 0.05, f"posterior-mean target should be ~useless: corr {c1:.3f}"
    assert c2 > 0.95, f"2nd-moment target should recover sufficiency: corr {c2:.3f}"


@pytest.mark.skip(reason="SCAFFOLD (S2 trained): cripple a flow's σ-context capacity "
                  "so ρ-average NF-MLE leaves a tail gap (μ₂-style), then retrain with "
                  "a minimax-over-θ-grid (or CVaR) reweighting; assert sup_θ coverage "
                  "error drops. The trained analog of test_s2_minimax_controls_sup_coverage.")
def test_s2_minimax_trained():
    ...


@pytest.mark.skip(reason="SCAFFOLD (§7 headline): 1-D two-component toy p(X|θ) whose "
                  "exact pivot gives a DISCONNECTED, exactly-covering C_α with no R1. "
                  "Validates that multimodality is a validity win (disconnected sets), "
                  "not a coverage failure. Needs a new BimodalLocation1D simulator.")
def test_validity_without_r1_disconnected_sets():
    ...
