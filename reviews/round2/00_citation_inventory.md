# Round 2 — Citation Inventory + BibTeX Migration Map

**Document under review:** `cd_sbi_v7.tex`
**Bibliography file:** `cd_sbi.bib`
**Bibkey convention:** AuthorYear (CamelCase). Two authors → both names
concatenated. Three or more → `EtAl`.

## Migration plan

The manuscript prior to round 2 uses inline text refs ("Hermans et
al.~(2022)"). Round 2 converts these to natbib `\cite` calls:
- `\citet{Key}` → "Author (Year)" — used in running prose where the
  author's name is the grammatical subject.
- `\citep{Key}` → "(Author, Year)" — used for parenthetical citations.
- `\cite{Key1, Key2}` → batched parenthetical when multiple refs cluster.

Preamble adds `\usepackage[round]{natbib}`. References section replaced
with `\bibliographystyle{plainnat}\bibliography{cd_sbi}`. Bibliography
entries not cited inline (Fraser2011, SinghEtAl2007, XieSingh2013,
CranmerEtAl2020, Lyons2013, SriperumbudurEtAl2011) are included via
`\nocite{}` so they appear in the rendered bibliography as
"related-literature" entries, matching the v6 reference list.

## Citation inventory

| Bibkey | First inline location | Other locations | Supporting claim (round-1 inventory ID) | Type |
|---|---|---|---|---|
| `HermansEtAl2022` | §1.1 (line ~56) | §1.1, §9 step 5, §11.6 | C-1.1-overconf, C-9-validate, OP-11.6 | primary |
| `PapamakariosEtAl2019` | abstract / §1.1 | §3.4, C-3.4-SNL row, §10 table | C-3.4-SNL, C-10-tbl | primary |
| `PapamakariosMurray2016` | §1.1 | §3.7 positioning, §10 table | C-3.7-positioning, C-10-tbl | primary |
| `GreenbergEtAl2019` | §1.1 | §3.7, §10 table | C-3.7-positioning, C-10-tbl | primary |
| `HermansEtAl2020` | §1.1 | §3.7, §10 table | C-3.7-positioning, C-10-tbl | primary |
| `MillerEtAl2022` | §1.1 | §10 table | C-10-tbl | primary |
| `DelaunoyEtAl2022` | §1.1 | §3.7, §10 table | C-3.7-positioning, C-10-tbl | primary |
| `FalkiewiczEtAl2023` | §1.1 | §3.7, §10 table, §11.6 | C-3.7-positioning, C-10-tbl, OP-11.6 | primary |
| `DalmassoEtAl2024` | §1.1 | §3.7 LF2I, §7.3 diagnostic, §10 table | C-3.7-CRPS (LF2I parenthetical), C-7.3-diagnostics, C-10-tbl | primary |
| `MasseranoEtAl2023` | §1.1 | §10 table | C-10-tbl | primary |
| `BortolatoVentura2025` | §1.1 | §10 table | C-10-tbl | primary |
| `PatelEtAl2023` | §1.1 | §10 table | C-10-tbl | primary |
| `Fisher1930` | §1.2 | — | D-CD (historical) | historical |
| `Cox1958` | §1.2 | §5.3 (ancillary def) | D-CD, C-5.3-tloc | historical |
| `Efron1993` | §1.2 | — | D-CD (historical) | historical |
| `Efron1998` | §1.2 | — | D-CD (historical) | historical |
| `SchwederHjort2016` | §1.2 | §4.5, §10 IRT | D-CD, P-4.5, C-10-IRT | textbook |
| `GneitingRaftery2007` | §3.2 | — | C-3.2-strictprop (def) | textbook |
| `WehenkelLouppe2019` | §3.5 | §7.1 D-UMNN | D-UMNN, C-9-arch-not-penalty | primary |
| `GrettonEtAl2005` | §3.7 glossary | — | C-3.7-Class2 (HSIC def) | methodological |
| `PapamakariosEtAl2021` | §6 intro | — | D-autoflow (NF review), OP-11.5 | review |
| `DurkanEtAl2019` | §6 intro | §11.5 | D-autoflow (NSF), OP-11.5 | primary |
| `HwangYang2001` | §5.7.3 | — | C-5.7-midp | primary |
| `Lancaster1961` | §5.7.1 | — | D-randPIT | historical |
| `Rosenblatt1952` | §6.3 (OT context) | — | T-A-d / KR (named after) | historical |
| `Brenier1991` | §6.3 (OT context) | — | T-A-d (Brenier comparison) | historical |
| `TaltsEtAl2018` | §7.3 | — | C-7.3-diagnostics (SBC) | methodological |
| `LemosEtAl2023` | §7.3 | — | C-7.3-diagnostics (TARP) | methodological |
| `CranmerEtAl2015` | §10 IRT | — | C-10-IRT | primary |
| `CarlierEtAl2010` | §11.2 | — | OP-11.2 (KR-Brenier interp) | methodological |
| `WehenkelEtAl2025` | §11.4 | — | OP-11.4 (related work) | primary |
| `SchmonEtAl2020` | §11.4 | — | OP-11.4 (robust SBI) | primary |
| `DellaportaEtAl2022` | §11.4 | — | OP-11.4 (robust SBI) | primary |
| `LueckmannEtAl2021` | §11.5 | §11.6 | OP-11.5, OP-11.6 | primary |
| `PapamakariosEtAl2017` | §11.5 | — | OP-11.5 (MAF; NF scale) | primary |

## Entries listed but not inline-cited (preserved via `\nocite`)

These appeared in the v6 References section without inline citations and
are kept as "see also" entries in the rendered bibliography.

| Bibkey | Topic | Reason for inclusion |
|---|---|---|
| `Fraser2011` | "Is Bayes posterior just quick and dirty confidence?" | Background for §1.2 CD vs posterior framing. |
| `SinghEtAl2007` | CD review | Background CD literature. |
| `XieSingh2013` | CD review | Background CD literature. |
| `CranmerEtAl2020` | SBI frontier (PNAS review) | Background SBI literature. |
| `Lyons2013` | Distance covariance | Background for §3.7 energy distance. |
| `SriperumbudurEtAl2011` | Characteristic kernels | Background for §3.7 HSIC / kernel propriety. |

## Round 2 lit-review focus per Part

Each round-2 critic agent gets the citation inventory + their assigned
Part of `cd_sbi_v7.tex`. Tasks per agent:

- Verify each cited paper's metadata (authors, year, venue) for the
  assigned Part is correct.
- Identify claims that should carry a citation but don't.
- Check whether the cited paper's contribution matches the manuscript's
  characterization (attribution accuracy).
- Write short paper notes at `references/<bibkey>.md` for each cited
  paper in the assigned Part.
- Surface any historical-context additions worth weaving into the .tex.

Part assignments per the round-1 split (Parts I–VII). Cross-Part flags
in the inventory:

- LF2I (`DalmassoEtAl2024`) appears in §1.1, §3.7, §7.3, §10. Part II
  and Part V agents both touch it; cross-reference via inventory.
- Hermans 2022 (`HermansEtAl2022`) appears in §1.1, §9, §11.6 — flagged
  in three Parts; align attribution across all.
- The §10 comparison table (`C-10-tbl`) bundles many citations into one
  Part VI claim; the Part VI agent should give it priority.
