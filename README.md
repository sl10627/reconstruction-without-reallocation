# Reconstruction Without Reallocation

**How did Chilean manufacturing capital adjust after the 2010 Maule earthquake?**

Emily (Sitong) Liu · New York University · Solo-authored working paper, 2026

---

The February 2010 Maule earthquake destroyed a large share of Chile's
manufacturing capital stock. I use it as an exogenous capital shock and ask
whether the reconstruction that followed *reallocated* factors of production, or
merely *replaced* what was destroyed.

Treatment is continuous: I overlay USGS ShakeMap intensity polygons onto GADM
administrative boundaries in an equal-area projection, which yields an
area-weighted Modified Mercalli Intensity for each region ranging from 2.9 to
6.7, with unshaken regions retained as controls at a floor value. I then estimate
Post × MMI difference-in-differences on 2-digit industry × region × year cells
from nineteen annual ENIA establishment databases (2003–2017), with industry ×
region and industry × year fixed effects.

**The result is a null with a pulse.** Investment rises sharply in 2014 and
2015, by 0.58 and 0.60 log points per intensity unit, before falling back.
Output, employment, labor productivity, the labor share, and profit proxies are
all indistinguishable from zero throughout. Capital was rebuilt, but nothing was
reallocated.

This rejects the capital–skill complementarity prediction of Krusell, Ohanian,
Ríos-Rull and Violante (2000) in this setting. I also rule out three competing
explanations for the investment pulse: construction-sector spillovers, unskilled
out-migration (checked against CASEN household data), and front-loading ahead of
the 2014 tax reform.

**On inference.** There are only fifteen region clusters, so I report a
restricted wild cluster bootstrap alongside the cluster-robust asymptotics, and
assess pre-trend sensitivity with Rambachan–Roth HonestDiD under both smoothness
and relative-magnitude restrictions. The breakdown values fall below the
conventional benchmark. I report them as a stated limitation rather than omitting
them.

---

## Repository map

| Path | Contents |
|---|---|
| `code/run_all.py` | Runs the whole pipeline in order, from the raw databases to the tables, then checks the paper's numbers |
| `code/01`–`code/34` | Python pipeline: extraction → harmonization → cell construction → GIS treatment → estimation → tables and figures |
| `code/verify_numbers.py` | Checks every number quoted in the paper text against fresh `output/*.csv` |
| `code/r/main_did.R` | `fixest` replication of the headline regressions, with a hand-rolled wild cluster bootstrap; checks itself against the Python estimates |
| `code/stata/main_did.do` | `reghdfe` + `boottest` replication of the same, with the same check |
| `code/README.md` | Full replication instructions, run order, and software requirements |
| `crosswalks/` | ISIC Rev. 3.1 → Rev. 4 concordance with activity-share weights |
| `paper/` | Manuscript source and compiled PDF |

## Data availability

**This repository contains code only.**

The analysis uses the *Encuesta Nacional Industrial Anual* (ENIA), collected by
Chile's Instituto Nacional de Estadísticas, together with CASEN household survey
data, USGS ShakeMap intensity rasters, and GADM administrative boundaries. The
ENIA databases are publicly available from INE but are not redistributed here.
See `code/README.md` for how to obtain them and where the scripts expect them to
sit.

One structural note that shapes the whole design: the public FUSION files from
2008 onward carry annual establishment identifiers that are not consistent across
years, so an establishment-level panel is not recoverable for the post-2007
period. Every regression therefore uses repeated cross-sections built from
industry × region × year cells.

## Reproducing the paper

```bash
pip install -r requirements.txt
# place the raw inputs where code/README.md lists them, then:
python code/run_all.py            # about 12 minutes, ends by checking every number in the paper
Rscript code/r/main_did.R         # optional: R replication, checked against Python
stata -b do code/stata/main_did.do  # optional: Stata replication, checked against Python
```

`code/17_make_tables.py` writes the LaTeX tables in `paper/tables/` directly from
the output CSVs, so the tables in the manuscript cannot drift away from the code.

## License

Code is released under the MIT License (see `LICENSE`). The manuscript in
`paper/` is © 2026 Emily (Sitong) Liu, all rights reserved.
