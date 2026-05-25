# Bortolato & Ventura, 2025 — Box Confidence Depth: Simulation-Based Inference with Hyper-Rectangles

**Authors:** Elena Bortolato, Laura Ventura
**Year:** 2025
**Venue:** arXiv preprint (stat.ME)
**arXiv:** 2502.11072
**DOI:** —

## One-paragraph summary
Proposes Box Confidence Depth (Box CD), an SBI method that constructs
frequentist confidence regions in the shape of *axis-aligned hyper-
rectangles* (boxes). The construction is based on a data-depth notion
adapted to the SBI setting: for each candidate `θ`, the depth of `X_obs`
in the simulator's predictive samples at `θ` is computed, and the
acceptance region is the smallest box containing the desired depth
level. The simulator-side calibration is performed via Monte Carlo
quantiles, in the spirit of LF2I but with a depth-based statistic
restricted to a hyper-rectangular geometry.

## Why CD-SBI cites it
- §1.1 "depth-based hyper-rectangles" — retrofit calibration family.
- §10 comparison-table row: target = "Confidence sets", calibration =
  "Depth-based, hyper-rectangles".

## Specific anchors
- Hyper-rectangle geometry (vs WALDO's ellipsoidal Wald regions or
  ACORE's likelihood-ratio level sets).
- Depth statistic for evidence ranking.

## Notes
- **Attribution-check OK.** arXiv 2502.11072 confirmed; first posted
  February 16, 2025 (with revisions through January 2026), so "2025"
  is the right year. Authors Bortolato and Ventura (Padova).
- This is the newest paper in the §10 table; no peer-reviewed venue
  yet, so the arXiv-only citation is appropriate.
