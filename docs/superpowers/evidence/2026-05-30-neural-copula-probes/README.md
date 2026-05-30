# Neural-copula exploration — evidence prototypes

Throwaway research probes behind the brainstorm notes
(`docs/superpowers/specs/2026-05-30-neural-copula-brainstorm-notes.md`). Not package
code, not test-covered. `PYTHONPATH=src python <file>` on a GPU box.

| file | what it shows |
|---|---|
| `proto_nle_score.py` | Probe #1: NLE (conditional MAF for p(X\|θ)) + Rao score pivot on (μ,Σ) d=5. Coverage 0.060 — escapes the summary bottleneck, recovers the covariance CD with no engineering. |
| `proto_nc_battery1.py` | Probes #1+#3 on (μ,Σ): calibrated score-norm (grid-free quantile head) → **0.038** (near oracle 0.026); full-latent χ²₂₀ pivot → 0.108 (inefficient). |
| `proto_nc_battery2b.py` | Probe #2 generality: Cauchy location-scale (NO sufficient statistic). asinh-stabilized NLE; score CD → **0.022** asymptotic / 0.046 calibrated — noise floor where the oracle/summary route is impossible. (Without asinh the MAF NaN'd — see notes finding #5.) |

Recipe: conditional flow q(X\|θ) + score-based d_θ CD (+ optional calibration). Scales
to arbitrary dim/distribution; no bottleneck; can't cheat. Asymptotic vs the
manuscript's exact KR pivot — complementary (notes §strategic fork).
