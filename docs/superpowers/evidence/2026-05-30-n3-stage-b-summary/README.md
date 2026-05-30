# N3 Stage-B learned-summary — evidence prototypes

Scratch prototypes behind the N3 verdict
(`docs/superpowers/specs/2026-05-30-cd-sbi-mu-cov-n3-stage-b-verdict.md`). Throwaway
research code (not part of the `cdsbi` package, not test-covered) — preserved for
reproducibility of the findings. Run with `PYTHONPATH=src python <file>` on a GPU box.

| file | what it shows |
|---|---|
| `proto_mu_cov_n3_riskgate.py` | Base I-A (AffineCoupling + ExactDensity) recovers means + logD₁₁ but collapses logD₂₂/D₂₁ while χ²₅ calibration passes. The original risk-gate. |
| `proto_mu_cov_n3_diagnose.py` | Capacity/routing/calibration probes: bigger+longer overfits *below* floor (cheat) and hurts recovery; info not hiding in ancillary; per-coord PIT bad at extremes. |
| `proto_mu_cov_n3_clean.py` | Clean fresh-batch (no overfit) read of the structural limit: 4/5, logD₂₂ stuck. |
| `proto_equivariant.py` | Naive permutation-equivariant coupling — scrambles, readout wrong, recovery *worse* (2/5). |
| `proto_asinh.py` | asinh-affine coupling (`sinh(a·asinh b+d)`) alone + with equivariance — unstable / inert; worse. |
| `proto_asinh_v3.py` | asinh conditioner-features (stable affine transform) + a no-overfit capacity control. **V4 (big plain affine, fresh) recovers cross-cov D₂₁→0.92.** |
| `proto_glow.py` | Glow invertible 1×1 (LU) channel-mixing — *more* flexibility → lazier 2/5 (means only). |
| `proto_v4_suff.py` | **Decisive** sufficiency analysis of the best variant: canonical corr `[.999 .999 .958 .919 .16]`, raw-moment R² (A₂₂=0.04) — one sufficient dim genuinely missing, not a parameterisation artifact. |
| `proto_struct.py` | **Existence proof:** Helmert+polar structure-informed invertible summary = exact Bartlett feats (diff 1e-5), invertible (1.9e-6), pivot calibrates (PIT≤0.069, χ²₅ 0.027). Bespoke → not productionised. |
| `proto_lf2i.py` | LF2I-BFF baseline at d=5, **UNFIXED** impl (N=64 box-grid marginal): coverage_error 0.20–0.39. Later confirmed an implementation artifact, not LF2I. |
| `proto_lf2i_fair.py` | LF2I-BFF at d=5 after the marginal fix (MC-from-true-prior, N=2048): coverage_error 0.091 (3-pt) / 0.189 (LHS) — the FAIR baseline. Decent central coverage, ~7× oracle worst-case. |

Verdict: generic *learned* summaries don't route the quadratic covariance stats;
flexibility hurts; bespoke fixes work but don't scale; LF2I-BFF fails too. → motivates
the scalable neural-copula direction (verdict §6).
