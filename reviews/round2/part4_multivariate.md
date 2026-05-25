# Round 2 — Part IV (Multivariate generalization) Lit-Review

## Summary

Part IV cites four works: `Rosenblatt1952` and `Brenier1991` for the
optimal-transport context of §6.3 (Knothe–Rosenblatt vs. Brenier maps),
and `PapamakariosEtAl2021` and `DurkanEtAl2019` at §6 intro as
standard references on triangular autoregressive flows. All four
bibkeys point to the right papers with correct metadata; the
characterizations in the manuscript are accurate. Two attribution
items are worth flagging: (i) the "Knothe–Rosenblatt" name in §6.3
currently cites only Rosenblatt 1952, but the compound name reflects
joint attribution to Knothe (1957, *Michigan Math. J.* 4, 39–52) who
independently arrived at the same triangular construction in
convex-bodies theory — this is the single most defensible historical-
context addition for Part IV; and (ii) §6 intro pairs Papamakarios
2021 (review) with Durkan 2019 (Neural Spline Flows) as references for
the *autoregressive triangular* pattern, but Durkan 2019's headline
contribution is the spline transformer block, not the autoregressive
structure itself — the natural primary-source citation for "autoregressive
triangular flow" is `PapamakariosEtAl2017` (MAF), currently cited only
in §11.5. No factual errors found; no §6.3 claim is materially
under-cited beyond the Knothe attribution.

## Per-citation findings

### `Rosenblatt1952` — Rosenblatt, *Annals Math. Stat.* 23(3), 470–472 (1952)
- **Metadata verified.** Author (Murray Rosenblatt), title ("Remarks on
  a multivariate transformation"), venue, vol/issue/pages, year all
  correct in `cd_sbi.bib`. DOI is `10.1214/aoms/1177729394` (Project
  Euclid); not in .bib but worth adding.
- **Attribution accuracy.** §6.3's use as the "triangular transport map
  named Knothe–Rosenblatt" is accurate for the Rosenblatt half of the
  joint attribution. Rosenblatt's three-page note proved that the
  triangular transform \(U_k = F_{k|<k}(X_k \mid X_{<k})\) pushes a
  continuous distribution to uniform on \([0,1]^d\). The CD-SBI pivot
  \(r_k^{\mathrm{KR}} = \Phi^{-1}(1 - F_k^{(\theta)}(X_k \mid X_{<k}))\)
  is exactly this transform composed with \(\Phi^{-1}\) (with sign-flip
  to match (R2\(^\mathrm{auto}\))).
- **Caveat.** Rosenblatt 1952 establishes the *transformation* but
  not a *uniqueness theorem* for triangular monotone maps between
  arbitrary measures; the modern uniqueness statement is folklore that
  Bonnotte 2013 and Santambrogio (*OT for Applied Mathematicians* §2.3)
  give clean proofs of. The manuscript's Theorem A-d does its own
  proof inductively, which is fine — it does not need to claim that the
  uniqueness statement itself comes from Rosenblatt 1952.

### `Brenier1991` — Brenier, *Comm. Pure Appl. Math.* 44(4), 375–417 (1991)
- **Metadata verified.** Author (Yann Brenier), title, journal,
  vol/no/pages, year all correct. DOI is `10.1002/cpa.3160440402`; not
  in .bib but worth adding.
- **Attribution accuracy.** §6.3's one-sentence characterization
  ("Brenier map ... is the unique transport map that is the gradient of
  a convex function (the \(L^2\)-optimal transport plan). KR depends
  on coordinate ordering; Brenier does not.") is accurate. Brenier
  1991 introduced the polar factorization \(u = (\nabla\phi) \circ s\)
  with \(\phi\) convex and \(s\) measure-preserving; the gradient map
  is the \(L^2\)-optimal Monge plan.
- **Companion citation note.** McCann (1995, *Duke Math. J.* 80(2),
  309–323, DOI `10.1215/S0012-7094-95-08013-2`) is the standard general-
  measure follow-up that relaxes Brenier's regularity assumptions
  (Brenier required absolute continuity + integrability; McCann needs
  only that the source give zero mass to every Lipschitz
  \((n-1)\)-surface). The manuscript does not need McCann for any
  load-bearing claim (Brenier is purely contrastive at §6.3), so this
  is an optional addition for completeness, not a gap.

### `DurkanEtAl2019` — Durkan, Bekasov, Murray, Papamakarios, NeurIPS 32 (2019)
- **Metadata verified.** Authors, title ("Neural spline flows"), venue
  (NeurIPS 2019, vol. 32), year all correct. arXiv 1906.04032; page
  range 7511–7522. Adding `pages` and `arxiv` fields would round out
  the entry.
- **Attribution accuracy — partial caveat.** §6 intro cites NSF as a
  standard reference on the *autoregressive triangular flow* pattern.
  NSF *does* include an autoregressive variant (NSF-AR), but the
  paper's headline contribution is the **monotonic rational-quadratic
  spline transformer**, applicable both to coupling layers (NSF-C) and
  autoregressive networks. The autoregressive scaffolding NSF-AR
  inherits is from `PapamakariosEtAl2017` (MAF). NSF is the right
  citation for "flexible monotone scalar transform inside an
  autoregressive layer"; for "autoregressive triangular structure
  itself" the primary source is MAF (currently only at §11.5). See the
  recommendation in Historical-context additions.

### `PapamakariosEtAl2021` — Papamakarios, Nalisnick, Rezende, Mohamed, Lakshminarayanan, *JMLR* 22 (2021)
- **Metadata verified.** Authors, title ("Normalizing flows for
  probabilistic modeling and inference"), JMLR vol. 22, pages 1–64,
  year all correct. arXiv 1912.02762.
- **Attribution accuracy.** §6 intro's framing — that this is a
  standard reference for the triangular autoregressive architectural
  pattern — is accurate. The review covers autoregressive flows with
  triangular Jacobians as one of its primary categories, alongside
  coupling, linear, residual, and continuous flows. It is the right
  go-to review citation.
- **Note.** The review does not, as far as I can verify, discuss the
  Knothe–Rosenblatt rearrangement under that name. So this is the
  right citation for the *deep-learning* framing of triangular
  autoregressive flows, not for the *measure-theoretic* uniqueness
  story (which is what §6.3 is doing). The two citation styles in §6
  intro (architectural pattern) and §6.3 (OT uniqueness) are
  appropriately distinct.

## Citation gaps

1. **Knothe (1957) is the missing half of "Knothe–Rosenblatt."**
   §6.3 names the map "Knothe–Rosenblatt rearrangement" but only cites
   Rosenblatt 1952. The compound name is standard because Knothe
   (Herbert Knothe, "Contributions to the theory of convex bodies,"
   *Michigan Math. J.* 4 (1957), 39–52) independently introduced the
   same triangular construction in convex-bodies / Brunn–Minkowski
   theory. The OT literature (Villani's *Topics in OT* and *OT: Old
   and New*; Santambrogio's *OT for Applied Mathematicians*) credits
   both. **Recommendation:** add `Knothe1957` to `cd_sbi.bib` and cite
   it alongside `Rosenblatt1952` at the first introduction of the
   Knothe–Rosenblatt name in §6.3. Suggested bib entry:
   ```bibtex
   @article{Knothe1957,
     author  = {Knothe, H.},
     title   = {Contributions to the theory of convex bodies},
     journal = {Michigan Mathematical Journal},
     volume  = {4},
     number  = {1},
     pages   = {39--52},
     year    = {1957},
     doi     = {10.1307/mmj/1028990175}
   }
   ```

2. **MAF (`PapamakariosEtAl2017`) belongs at §6 intro, not just §11.5.**
   The §6 intro paragraph attributes the autoregressive-triangular
   architectural pattern to Papamakarios 2021 (review) and Durkan 2019
   (NSF). The primary source for autoregressive normalizing flows with
   triangular Jacobian is MAF (Papamakarios, Pavlakou, Murray, NeurIPS
   2017) — which the inventory already lists, just for §11.5 only.
   **Recommendation:** at §6 intro, cite `{PapamakariosEtAl2017,
   PapamakariosEtAl2021, DurkanEtAl2019}` together — MAF as the primary
   source for the architectural pattern, the survey as the textbook
   reference, NSF as the modern flexible transformer block. This is a
   light edit (one citation added at one location) and improves the
   primary-source pedigree of the §6 intro.

3. **Theorem A-d's uniqueness within the autoregressive class — no
   additional citation needed.** §6.3 proves uniqueness inductively
   from first principles (Lemma 4.2 applied at each level + sigma-
   algebra equivalence). Folklore OT uniqueness statements for the
   triangular monotone map exist (see Bonnotte 2013 and Carlier–
   Galichon–Santambrogio 2010), but they target a different setting
   (transport between two fixed measures) — Theorem A-d's setting
   (transport from the family \(\{X \mid \theta\}_\theta\) to
   \(\mathcal{N}(0, I_d)\) with the (R1\(^\mathrm{auto}\))/
   (R2\(^\mathrm{auto}\)) constraints) is genuinely new. The
   manuscript's existing forward-reference to `CarlierEtAl2010` in §11.2
   (KR-Brenier continuation) is the right home for that citation.

## Historical-context additions

In priority order:

1. **Knothe 1957** at §6.3 — the only addition I think is materially
   important. Without it, the compound name "Knothe–Rosenblatt" is
   half-attributed. Add `Knothe1957` to the .bib, cite as
   `\citep{Knothe1957, Rosenblatt1952}` at the first occurrence of the
   name in §6.3 paragraph 1.

2. **McCann 1995** at §6.3 — optional. If the manuscript wants to
   round out the Brenier sentence to acknowledge the general-measure
   form, a parenthetical "(extended to general measures by
   \citealp{McCann1995})" after the `\citep{Brenier1991}` citation
   would do. Not load-bearing; only adds completeness.

3. **Santambrogio 2015** (*OT for Applied Mathematicians*) or
   **Villani 2003/2009** as a one-stop reference for the OT context —
   optional, but if §6.3 ever expands the OT remarks beyond the one
   paragraph, one of these textbooks is the standard citation for
   anyone wanting more depth. Not needed now.

Item (1) is the only one I'd flag as a near-required edit for round 2.
Items (2)–(3) are nice-to-haves.
