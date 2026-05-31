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

## Probing round 2 (scaling + benchmark + comparison)

| file | what it shows |
|---|---|
| `proto_nc_battery3.py` | Higher-d (d_θ=10, 5×Gaussian(μ,logσ)): score 0.058 — scales. SLCP (d_θ=5 benchmark): 0.115 asymptotic / 0.426 calibrated — BREAKS (multimodal + Fisher-degenerate; the regularity boundary). |
| `proto_nc_battery4.py` | NC on the previous test cases vs known CD-SBI: §8.2/8.3 ~0.03–0.04, §8.4 0.06, (μ,σ²) 0.030 calibrated (vs 0.026). (§8.1 here is a d_x=1 zero-padding artifact 0.227; clean features=1 run is 0.009.) |

## Probing round 3 (NSF density model; method named Score-CD)

| file | what it shows |
|---|---|
| `proto_score_cd_zuko.py` | Score-CD with a zuko **NSF** density model (GPU-efficient). SLCP: NSF fit far better (-logq 10.99→3.06) but coverage UNCHANGED (0.111≈0.115), calibrated worse (0.648) → SLCP breakage is intrinsic regularity (Fisher degeneracy + sign-symmetry), NOT density quality. (μ,Σ): NSF≈MAF (0.066 vs 0.060). MAF→NSF is not a coverage lever; NSF is the sensible default density model regardless. |
