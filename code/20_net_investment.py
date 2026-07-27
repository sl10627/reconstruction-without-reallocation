"""Net vs timing: did the reconstruction pulse leave NET capital accumulation, or
was it front-loaded investment that reverted (pure replacement timing)?

Investment is a flow; the net stock change is cumulative investment. On the
pre-period-projected detrended residual (whose pre-period mean is ~0 by
construction), a single Post x MMI coefficient measures the AVERAGE annual
post-2010 deviation from the pre-trend baseline:
  beta_post > 0  -> net accumulation (flow ran above baseline on average; stock rose)
  beta_post ~ 0  -> pure timing (front-loaded spike offset by later undershoot)
I also split early (2010-2014, rebuilding) vs late (2015-2017, after the peak).

Run for total investment, structures, and equipment.

Output: output/net_investment_test.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"

MAIN_IND2 = ["10", "16", "24"]
MMI_FLOOR = 2.0
W = "weight_sum"
PRE_YEARS = [2003, 2004, 2005, 2006, 2007, 2008, 2009]


def _sum(df, cols):
    cc = [c for c in cols if c in df.columns]
    return df[cc].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1) if cc else pd.Series(np.nan, index=df.index)


def build():
    p = pd.read_parquet(DATA / "enia_pooled.parquet")
    p = p[p["ind_rev4"].notna() & p["REGION"].notna() & p["ANIO"].notna()].copy()
    p["ind2"] = p["ind_rev4"].astype(str).str[:2]
    p["REGION"] = pd.to_numeric(p["REGION"], errors="coerce")
    p["ANIO"] = pd.to_numeric(p["ANIO"], errors="coerce").astype(int)
    p["w"] = pd.to_numeric(p["ind_weight"], errors="coerce").fillna(0.0)
    old = p["ANIO"] <= 2014
    p["total"] = np.where(old, _sum(p, ["CBNTER", "CBNEDI", "CBNMAQ", "CBNVEH", "CBNMUE", "CBNSOF"]),
                          _sum(p, ["CBTER", "CBEDI", "CBMAQ", "CBVEH", "CBSOF"]))
    p["structures"] = np.where(old, _sum(p, ["CBNEDI", "CBNTER"]), _sum(p, ["CBEDI", "CBTER"]))
    p["equipment"] = np.where(old, _sum(p, ["CBNMAQ", "CBNVEH", "CBNSOF"]), _sum(p, ["CBMAQ", "CBVEH", "CBSOF"]))

    def wsum(v, w):
        m = pd.notna(v) & (w > 0)
        return (v[m] * w[m]).sum() if m.any() else np.nan

    rows = []
    for (i2, reg, yr), g in p.groupby(["ind2", "REGION", "ANIO"]):
        rows.append({"ind2": i2, "REGION": int(reg), "ANIO": int(yr), "weight_sum": g["w"].sum(),
                     **{k: wsum(g[k], g["w"]) for k in ["total", "structures", "equipment"]}})
    c = pd.DataFrame(rows)
    sh = pd.read_csv(DATA / "shakemap_by_region.csv").rename(columns={"enia_region": "REGION"})
    c = c.merge(sh[["REGION", "mean_mmi"]], on="REGION", how="left")
    c["mean_mmi"] = c["mean_mmi"].fillna(MMI_FLOOR)
    c["year_c"] = c["ANIO"] - 2009
    c["jr"] = c["ind2"] + "_r" + c["REGION"].astype(str)
    c["jt"] = c["ind2"] + "_t" + c["ANIO"].astype(str)
    for k in ["total", "structures", "equipment"]:
        c[f"log_{k}"] = np.where(c[k] > 0, np.log(c[k]), np.nan)
    return c


def detrend_pre(cell, ycol):
    sub = cell[cell["ind2"].isin(MAIN_IND2)].dropna(subset=[ycol]).copy()
    for k in ["jr", "jt"]:
        vc = sub[k].value_counts()
        sub = sub[sub[k].isin(vc[vc >= 2].index)]
    pre = sub[sub["ANIO"].isin(PRE_YEARS)]
    jr_mean = (pre.groupby("jr").apply(lambda d: (d[ycol] * d[W]).sum() / d[W].sum(),
                                       include_groups=False).to_dict())
    sub = sub[sub["jr"].isin(jr_mean)].copy()
    sub["_dm"] = sub[ycol] - sub["jr"].map(jr_mean)
    slopes = {}
    for r, d in sub[sub["ANIO"].isin(PRE_YEARS)].groupby("REGION"):
        slopes[r] = (sm.WLS(d["_dm"], sm.add_constant(d["year_c"]), weights=d[W].clip(lower=1)).fit()
                     .params.get("year_c", 0.0)) if (d["year_c"].nunique() >= 2 and len(d) >= 3) else 0.0
    sub["_detr"] = sub["_dm"] - sub["REGION"].map(slopes).fillna(0.0) * sub["year_c"]
    return sub


def main():
    cell = build()
    rows = []
    for col, lab in [("log_total", "Total investment"), ("log_structures", "Structures"),
                     ("log_equipment", "Equipment")]:
        sub = detrend_pre(cell, col)
        sub["post_mmi"] = (sub["ANIO"] >= 2010) * sub["mean_mmi"]
        sub["early_mmi"] = sub["ANIO"].between(2010, 2014) * sub["mean_mmi"]
        sub["late_mmi"] = (sub["ANIO"] >= 2015) * sub["mean_mmi"]
        # average post deviation
        m1 = smf.wls("_detr ~ post_mmi + C(jt)", data=sub, weights=sub[W].clip(lower=1)).fit(
            cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
        # early vs late
        m2 = smf.wls("_detr ~ early_mmi + late_mmi + C(jt)", data=sub, weights=sub[W].clip(lower=1)).fit(
            cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
        rows.append({"capital": lab,
                     "avg_post": m1.params["post_mmi"], "avg_post_se": m1.bse["post_mmi"], "avg_post_p": m1.pvalues["post_mmi"],
                     "early_2010_14": m2.params["early_mmi"], "early_p": m2.pvalues["early_mmi"],
                     "late_2015_17": m2.params["late_mmi"], "late_p": m2.pvalues["late_mmi"], "n": int(m1.nobs)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "net_investment_test.csv", index=False)
    print("=== Net vs timing: average post-2010 deviation from pre-trend (per MMI) ===")
    print(df.round(4).to_string(index=False))
    print("\nReading: avg_post>0 & significant -> net accumulation; "
          "late << early -> front-loaded/reverting.")


if __name__ == "__main__":
    main()
