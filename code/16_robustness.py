"""Robustness of the headline 2014-2015 reconstruction INVESTMENT pulse.

Baseline (from 15_earthquake_did.py, pre-period-projected detrending):
    2014 investment x MMI = +0.561 (p=0.001), 2015 = +0.576 (p=0.022)

Each variant re-estimates the pulse with the SAME pre-period-projected method
(region/cell linear trend estimated on pre-quake years only, projected forward,
event study on the residual), changing one design choice at a time:

  baseline         MAIN (Food/Wood/Metals), continuous MMI, region pre-trend, 2003-2017
  R1 ind x region  trend estimated per industry x region cell (more flexible)   <-- requested
  R2 tight window  restrict to 2007-2017 (pre = 2007-2009, no long extrapolation) <-- requested
  R3 drop Santiago drop region 13 (largest urban region) -> not driven by capital
  R4 binary MMI    treatment = 1[MMI>=6] instead of continuous (functional form)
  R5 all manuf     full manufacturing sample instead of 3 clean industries

Outputs:
  output/earthquake_robustness_pulse.csv
  output/figs/E6_robustness_pulse.png   forest plot of the 2014 pulse across variants
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
FIGS = OUT / "figs"

MAIN_IND2 = ["10", "16", "24"]
W = "weight_sum"


def _fe_ok(sub: pd.DataFrame, ycol: str) -> pd.DataFrame:
    sub = sub.dropna(subset=[ycol]).copy()
    for key in ["jr", "jt"]:
        c = sub[key].value_counts()
        sub = sub[sub[key].isin(c[c >= 2].index)]
    return sub


def pulse(cell: pd.DataFrame, *, ycol="log_inv", treat="mean_mmi", trend_by="REGION",
          inds=MAIN_IND2, min_year=2003) -> dict:
    """Pre-period-projected event study; return 2014 & 2015 treat-x-year coefs."""
    pre_years = [y for y in [2003, 2004, 2005, 2006, 2007, 2008, 2009] if y >= min_year]
    es_years = [y for y in [2003, 2004, 2005, 2006, 2007, 2008, 2010, 2011, 2013, 2014, 2015, 2016, 2017]
                if y >= min_year]  # 2009 = reference
    sdf = cell[cell["ANIO"] >= min_year].copy()
    if inds is not None:
        sdf = sdf[sdf["ind2"].isin(inds)]
    sub = _fe_ok(sdf, ycol)
    pre = sub[sub["ANIO"].isin(pre_years)]
    if pre.empty or sub["REGION"].nunique() < 3:
        return {"beta_2014": np.nan, "se_2014": np.nan, "p_2014": np.nan,
                "beta_2015": np.nan, "p_2015": np.nan, "n": len(sub)}

    # (1) remove industry x region pre-period level
    jr_mean = (pre.groupby("jr").apply(lambda d: (d[ycol] * d[W]).sum() / d[W].sum(),
                                       include_groups=False).to_dict())
    sub = sub[sub["jr"].isin(jr_mean)].copy()
    sub["_dm"] = sub[ycol] - sub["jr"].map(jr_mean)

    # (2) per-unit (region or industry x region) linear slope, estimated on pre only
    slopes = {}
    for key, d in sub[sub["ANIO"].isin(pre_years)].groupby(trend_by):
        if d["year_c"].nunique() >= 2 and len(d) >= 3:
            mm = sm.WLS(d["_dm"], sm.add_constant(d["year_c"]), weights=d[W].clip(lower=1)).fit()
            slopes[key] = mm.params.get("year_c", 0.0)
        else:
            slopes[key] = 0.0
    sub["_detr"] = sub["_dm"] - sub[trend_by].map(slopes).fillna(0.0) * sub["year_c"]

    # (3) event study on residual
    for y in es_years:
        sub[f"t_{y}"] = sub[treat] * (sub["ANIO"] == y).astype(int)
    terms = " + ".join(f"t_{y}" for y in es_years)
    m = smf.wls(f"_detr ~ {terms} + C(jt)", data=sub, weights=sub[W].clip(lower=1)).fit(
        cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
    return {"beta_2014": m.params.get("t_2014", np.nan), "se_2014": m.bse.get("t_2014", np.nan),
            "p_2014": m.pvalues.get("t_2014", np.nan), "beta_2015": m.params.get("t_2015", np.nan),
            "p_2015": m.pvalues.get("t_2015", np.nan), "n": int(m.nobs)}


def main():
    cell = pd.read_parquet(DATA / "enia_cell_2digit_quake.parquet")

    variants = [
        ("baseline (MAIN, region trend)",      dict()),
        ("R1 industry x region trends",        dict(trend_by="jr")),
        ("R2 tight window 2007-2017",          dict(min_year=2007)),
        ("R3 drop Santiago (reg 13)",          dict()),   # cell filtered below
        ("R4 binary MMI>=6",                   dict(treat="high_mmi")),
        ("R5 all manufacturing",               dict(inds=None)),
    ]
    rows = []
    for name, kw in variants:
        c = cell[cell["REGION"] != 13] if name.startswith("R3") else cell
        r = pulse(c, **kw)
        r["variant"] = name
        rows.append(r)
        print(f"{name:32s} 2014 beta={r['beta_2014']:+.3f} (se {r['se_2014']:.3f}, "
              f"p {r['p_2014']:.3f}) | 2015 beta={r['beta_2015']:+.3f} (p {r['p_2015']:.3f}) | n={r['n']}")

    tbl = pd.DataFrame(rows)[["variant", "beta_2014", "se_2014", "p_2014", "beta_2015", "p_2015", "n"]]
    tbl.to_csv(OUT / "earthquake_robustness_pulse.csv", index=False)
    print(f"\nWrote {OUT / 'earthquake_robustness_pulse.csv'}")

    # forest plot of the 2014 pulse
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(tbl))[::-1]
    ax.errorbar(tbl["beta_2014"], y, xerr=1.96 * tbl["se_2014"], fmt="o",
                color="darkred", capsize=4)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(tbl["variant"], fontsize=9)
    ax.set_xlabel("2014 investment x MMI coefficient (95% CI)")
    ax.set_title("Robustness of the 2014 reconstruction-investment pulse\n"
                 "pre-period-projected detrending; one design change per row", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "E6_robustness_pulse.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {FIGS / 'E6_robustness_pulse.png'}")


if __name__ == "__main__":
    main()
