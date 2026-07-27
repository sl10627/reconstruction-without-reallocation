# Replication package — *Reconstruction Without Reallocation*

This package reproduces every table and figure in "Reconstruction Without
Reallocation: How Did Chilean Manufacturing Capital Adjust After the 2010 Maule
Earthquake Shock?" I treat the Maule earthquake as an exogenous shock to Chile's
manufacturing sector and estimate its effect with a difference-in-differences
design across industries and regions, using ENIA establishment data and USGS
ShakeMap intensity.

All of the tables and figures in the paper are produced by the replication
scripts, which go from the raw annual databases through extraction,
harmonization, cell construction, intensity merging, estimation, and table and
figure generation. I wrote the full pipeline in Python (`code/01`–`code/31`).
For reviewers who work in Stata or R, I also provide a focused replication of the
headline regressions in `code/stata/` and `code/r/`.

---

## 1. Software

- **Python** 3.11+ with `pandas`, `numpy`, `statsmodels`, `geopandas`,
  `pyarrow`, and `matplotlib`. Nothing in the paper is entered by hand; every
  number comes from these scripts.
- **R** 4.x with `fixest`, for `code/r/main_did.R`. I read the data with base
  `read.csv`, so `arrow` is not needed. `fwildclusterboot` is optional. It is
  not available for R ≥ 4.6, so I hand-roll the wild cluster bootstrap in the
  script instead.
- **Stata** 15+ with `reghdfe`, `ftools`, and `boottest`
  (`ssc install reghdfe ftools boottest`), for `code/stata/main_did.do`.

## 2. Data

This paper is based on data from the Annual National Industrial Survey
(*Encuesta Nacional Industrial Anual*, ENIA) collected by the Chilean National
Institute of Statistics (INE). The raw ENIA databases (`*-base-enia.accdb`,
2000–2018) sit in the project root. CASEN and the earthquake geodata are under
`data/` and `geodata/`. The ENIA databases are publicly available from INE but
are not redistributed here; the scripts expect the `.accdb` files at the
project root.

The publicly available FUSION files from 2008 onward use annual identifiers that
are not consistent across years; I therefore do not construct an
establishment-level panel for the period after 2007. Instead, all regressions
use repeated cross-sections composed of "2-digit industry × region × year"
cells.

## 3. Reproducing the paper (Python)

I run the scripts in order. Steps 01–06 build the panel from the raw `.accdb`
files; from 13 on they estimate. The working sample covers the years 2003–2017,
excluding 2012 because the establishment response rate to the questionnaire was
markedly lower than in other years.

```
04_harmonize.py            # raw PARTE csv -> wide_<year>.parquet
05_build_crosswalk_weights.py
06_build_pooled.py         # -> data/enia_pooled.parquet
13_earthquake_intensity.py # -> data/shakemap_by_region.csv, Figure 1
31_remake_fig1.py          # redraw Figure 1 with accented region labels
15_earthquake_did.py       # main DiD and event studies
16_robustness.py           # robustness of the 2014 pulse
18–21                      # labor, asset decomposition, net investment, wages
22_wild_bootstrap.py       # wild cluster bootstrap (fixed seed)
24_honestdid.py            # Rambachan–Roth sensitivity
25_casen_migration.py      # out-migration cross-check (CASEN)
27_casen_construction.py   # construction-sector placebo
29_incidence.py            # incidence decomposition
30_spillover.py            # SLX spillover test
17_make_tables.py          # writes paper/tables/*.tex from the output CSVs
```

The numeric outputs land in `output/*.csv`, and `17_make_tables.py` writes the
LaTeX tables in `paper/tables/` directly from those CSVs, so the tables in the
paper cannot drift from the code.

## 4. Checking that the paper matches the code

```
python code/verify_numbers.py
```

I use this to check every hard-coded number in the paper prose against the fresh
`output/*.csv`. It exits 0 when every number matches, and otherwise prints the
claim that is out of sync together with its paper-vs-pipeline values. I run it
after changing any regression.

## 5. Stata and R replication of the main results

Both scripts read one exported panel (`data/analysis_cell.csv`, written from
`data/enia_cell_2digit_quake.parquet`; I also provide a `.dta` copy) and
reproduce the four numbers the paper leans on:

| # | Specification | β (2014) |
|---|---|---|
| 1 | main DiD, investment, no trend | +0.517 |
| 2 | main DiD, investment, + region linear trend | +0.168 |
| 3 | raw event study, investment, 2014 | +0.712 |
| 4 | pre-period-projected detrend, 2014 (**headline**) | +0.579 |

```
Rscript code/r/main_did.R          # fixest::feols, plus a hand-rolled wild bootstrap
stata -b do code/stata/main_did.do # reghdfe + boottest
```

For the headline specification, I estimate each region's linear trend using only
the pre-earthquake years (2003 through 2009) and project it forward, then run the
event study on the deviations from this pre-period-projected path. I do not
estimate the trend on the whole sample, because that would let the
post-earthquake reconstruction pulse affect the counterfactual slope and pull the
treatment effect toward zero. Both scripts follow this same procedure.

I have run the R script, and it matches the Python pipeline to the fourth
decimal, including the bootstrap *t*-statistic (3.40). The Stata `.do` mirrors
the same logic with `reghdfe` and `boottest`.

Because there are only fifteen region clusters, I read the cluster-robust
asymptotic standard errors cautiously, and for the headline pulse I also report a
restricted wild cluster bootstrap *p*-value with Rademacher weights. This
bootstrap *p*-value moves a little with the random draws (Python ≈ 0.027,
R ≈ 0.034). The point estimate and the *t*-statistic are identical across
Python, R, and Stata.
