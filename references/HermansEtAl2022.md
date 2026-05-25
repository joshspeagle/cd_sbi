# Hermans et al., 2022 — A trust crisis in simulation-based inference? Your posterior approximations can be unfaithful

**Authors:** Joeri Hermans, Arnaud Delaunoy, François Rozet, Antoine Wehenkel, Volodimir Begy, Gilles Louppe
**Year:** 2022
**Venue:** Transactions on Machine Learning Research (TMLR)
**arXiv:** 2110.06581
**DOI:** n/a (TMLR; OpenReview forum id `LHAbHkt6Aq`)

## One-paragraph summary
A large empirical study of the dominant neural-SBI estimators — (S)NPE, (S)NRE, and ABC variants — testing how reliably their reported credible regions cover the true parameter under realistic simulation budgets. The headline finding is that all benchmarked methods can produce systematically overconfident posterior approximations: empirical coverage of nominal-α credible regions falls well below α, often dramatically so, across most tested tasks and budgets. The paper proposes ensembling and "conservative" training as partial mitigations but frames the broader concern as a calibration crisis for scientific use of SBI.

## Why CD-SBI cites it
Anchor for the §1.1 problem statement (claim ID **C-1.1-overconf**): the central motivation for proposing a frequentist CD target rather than a Bayesian posterior. Also referenced in §9 step 5 (validate coverage) and §11.6 (open problem on sequential / amortized variants).

## Specific anchors
- Abstract + Figs. 1–3 (the coverage-vs-nominal plots across tasks) for the overconfidence finding.
- The benchmark spans (S)NPE, (S)NRE, and ABC variants. The CD-SBI manuscript's phrase "all four major SBI families" is slightly loose — NLE/SNL is not explicitly in the Hermans et al. benchmark, though the conclusion is widely understood to extend to it. Worth a softening in §1.1 (see report).

## Notes
- TMLR papers do not carry traditional volume/issue/page numbers; the arXiv id and OpenReview forum id are the canonical handles.
- Attribution caveat above: manuscript says "all four major SBI families" but the paper actually benchmarks three (NPE, NRE, ABC families). Minor wording issue, not a misattribution.
