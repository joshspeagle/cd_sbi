# Lancaster, 1961 — Significance Tests in Discrete Distributions

**Authors:** H. O. Lancaster
**Year:** 1961
**Venue:** Journal of the American Statistical Association, 56(294), 223–234
**DOI:** 10.1080/01621459.1961.10482105 (JSTOR: https://www.jstor.org/stable/2282247)

## One-paragraph summary

Lancaster reviews the long-standing problem that, for discrete test
statistics, the standard p-value is conservative — its null distribution
is stochastically larger than Uniform(0, 1) — and surveys two
remedies. The first is auxiliary randomization (smear each atom across
an interval of an independent `U ~ Uniform(0, 1)`), which Lancaster
notes is the classical fix going back at least to Stevens (1950) and
Tocher (1950) for making the rejection probability exactly hit the
nominal level. The second is the **mid-p value** (Lancaster's main
proposal in this paper), defined as half the probability of the
observed atom plus the probability of more extreme values. Lancaster
argues the mid-p is a defensible deterministic compromise: roughly
calibrated, not actually random, and easier to defend in practice
than the randomized procedure.

## Why CD-SBI cites it

CD-SBI cites Lancaster 1961 at §5.7.1 as the source for the
randomized-PIT construction
`V_θ(X, U) = F_θ(T(X)^-) + U · p_θ(T(X))`, which is the basis for the
discrete extension of the framework (Theorem C*, §5.7.2), and at
§5.7.3 (via `HwangYang2001`) as the originator of the mid-p value
that gives the deterministic `U = 1/2` variant.

## Specific anchors

- **Randomized PIT formula.** Lancaster discusses auxiliary
  randomization but is not, strictly speaking, the originator of the
  `V = F(t^-) + U · p(t)` construction; that form is widely credited
  to Stevens (1950, *Biometrika*) and Tocher (1950, *Biometrika*) in
  the secondary literature (see Hannig et al. 2016; Taraldsen &
  Lindqvist 2013). Lancaster's own contribution here is the review
  + the mid-p proposal, not the randomized PIT per se.
- **Mid-p value.** Lancaster *is* the originator of the mid-p
  significance level for discrete tests; this is the attribution that
  is uncontested and is what Hwang & Yang (2001) explicitly cite
  Lancaster for ("the mid p-value was proposed first by Lancaster
  (1961)").

## Notes

- Attribution audit: §5.7.1 cites Lancaster for "auxiliary
  randomization: smear each atom across an interval of an auxiliary
  uniform variable." This is partially correct — Lancaster discusses
  the construction — but the priority for the precise
  `V = F(t^-) + U · p(t)` formula sits with Stevens 1950 (and Tocher
  1950 for the parallel testing-theory development). A more accurate
  attribution would be `\citep{Stevens1950, Lancaster1961}` or a
  footnote acknowledging Stevens' priority. This is a soft flag, not
  a correctness issue: Lancaster 1961 is a defensible secondary
  reference and is the one Hwang & Yang use.
- The mid-p attribution at §5.7.3 (via `HwangYang2001`) is correct.
