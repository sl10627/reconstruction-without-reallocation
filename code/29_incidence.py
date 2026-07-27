"""Incidence of the capital shock: did the returns go to capital or labor?

The reconstruction-investment pulse is the one margin that moves; output, employment,
and wages do not. This script asks the distribution question directly, on the same
pre-period-projected-detrended footing, MAIN sample:

  log REMPAG        total wage bill (labor compensation)        all years
  log VBP           gross output                                all years
  labor_share       REMPAG / VBP (labor's share of gross output) all years
  log RENIMP        corporate income tax (profit proxy)*         all years
  log investment    CBN/CB (reference -- the margin that moves)

Interpretation. If the shock's incidence fell on CAPITAL rather than labor, we expect:
investment up, but the wage bill and labor's share of output NOT up (labor does not
capture more of the pie), and the profit proxy not falling. VA-based labor shares are
infeasible (VA is only available 2003-07 and 2014+), so labor share is measured on gross
output. RENIMP is contaminated by the 2014 rate increase (absorbed by industry x year FE)
and by investment-linked depreciation, so it is a noisy, secondary proxy.

Output: output/incidence_did.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "output"
W = "weight_sum"
MAIN = ["10", "16", "24"]
MMI_FLOOR = 2.0
PRE = [2003, 2004, 2005, 2006, 2007, 2008, 2009]
CBN = ["CBNTER", "CBNEDI", "CBNMAQ", "CBNVEH", "CBNMUE", "CBNSOF"]
CB = ["CBTER", "CBEDI", "CBMAQ", "CBVEH", "CBSOF"]


def _sum(df, cols):
    cc = [c for c in cols if c in df.columns]
    return df[cc].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)


def build():
    p = pd.read_parquet(DATA / "enia_pooled.parquet")
    p = p[p["ind_rev4"].notna() & p["REGION"].notna() & p["ANIO"].notna()].copy()
    p["ind2"] = p["ind_rev4"].astype(str).str[:2]
    p["REGION"] = pd.to_numeric(p["REGION"], errors="coerce")
    p["ANIO"] = pd.to_numeric(p["ANIO"], errors="coerce").astype(int)
    p["w"] = pd.to_numeric(p["ind_weight"], errors="coerce").fillna(0.0)
    p["rempag"] = pd.to_numeric(p["REMPAG"], errors="coerce")
    p["vbp"] = pd.to_numeric(p["VBP"], errors="coerce")
    p["renimp"] = pd.to_numeric(p["RENIMP"], errors="coerce")
    p["emp"] = pd.to_numeric(p["EMPTOT"], errors="coerce")
    p["inv"] = np.where(p["ANIO"] <= 2014, _sum(p, CBN), _sum(p, CB))

    def wsum(v, w):
        m = pd.notna(v) & (v > 0) & (w > 0)
        return (v[m] * w[m]).sum() if m.any() else np.nan

    rows = []
    for (i2, reg, yr), g in p.groupby(["ind2", "REGION", "ANIO"]):
        w = g["w"]
        rows.append({"ind2": i2, "REGION": int(reg), "ANIO": int(yr), "weight_sum": w.sum(),
                     "rempag": wsum(g["rempag"], w), "vbp": wsum(g["vbp"], w),
                     "renimp": wsum(g["renimp"], w), "inv": wsum(g["inv"], w),
                     "emp": wsum(g["emp"], w)})
    c = pd.DataFrame(rows)
    sh = pd.read_csv(DATA / "shakemap_by_region.csv").rename(columns={"enia_region": "REGION"})
    c = c.merge(sh[["REGION", "mean_mmi"]], on="REGION", how="left")
    c["mean_mmi"] = c["mean_mmi"].fillna(MMI_FLOOR)
    c["year_c"] = c["ANIO"] - 2009
    c["jr"] = c["ind2"] + "_r" + c["REGION"].astype(str)
    c["jt"] = c["ind2"] + "_t" + c["ANIO"].astype(str)
    c["labor_share"] = c["rempag"] / c["vbp"]
    c["lprod"] = np.where((c["vbp"] > 0) & (c["emp"] > 0), c["vbp"] / c["emp"], np.nan)
    c["log_rempag"] = np.log(c["rempag"])
    c["log_vbp"] = np.log(c["vbp"])
    c["log_renimp"] = np.log(c["renimp"])
    c["log_inv"] = np.log(c["inv"])
    c["log_emp"] = np.where(c["emp"] > 0, np.log(c["emp"]), np.nan)
    c["log_lprod"] = np.where(c["lprod"] > 0, np.log(c["lprod"]), np.nan)
    c["log_lshare"] = np.where(c["labor_share"] > 0, np.log(c["labor_share"]), np.nan)
    return c


def avgpost(c, col):
    sub = c[c["ind2"].isin(MAIN)].dropna(subset=[col]).copy()
    for k in ["jr", "jt"]:
        vc = sub[k].value_counts(); sub = sub[sub[k].isin(vc[vc >= 2].index)]
    pre = sub[sub["ANIO"].isin(PRE)]
    jrm = pre.groupby("jr").apply(lambda d: (d[col] * d[W]).sum() / d[W].sum(),
                                  include_groups=False).to_dict()
    sub = sub[sub["jr"].isin(jrm)].copy()
    sub["dm"] = sub[col] - sub["jr"].map(jrm)
    sl = {}
    for r, d in sub[sub["ANIO"].isin(PRE)].groupby("REGION"):
        sl[r] = (sm.WLS(d["dm"], sm.add_constant(d["year_c"]), weights=d[W].clip(lower=1)).fit()
                 .params.get("year_c", 0.0)) if (d["year_c"].nunique() >= 2 and len(d) >= 3) else 0.0
    sub["detr"] = sub["dm"] - sub["REGION"].map(sl).fillna(0.0) * sub["year_c"]
    sub["post_mmi"] = (sub["ANIO"] >= 2010) * sub["mean_mmi"]
    m = smf.wls("detr ~ post_mmi + C(jt)", data=sub, weights=sub[W].clip(lower=1)).fit(
        cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
    return m.params["post_mmi"], m.bse["post_mmi"], m.pvalues["post_mmi"], int(m.nobs)


def main():
    c = build()
    rows = []
    for col, lab in [("log_inv", "Investment"),
                     ("log_vbp", "Gross output (VBP)"),
                     ("log_emp", "Employment"),
                     ("log_lprod", "Labor productivity (VBP/worker)"),
                     ("log_rempag", "Wage bill (REMPAG)"),
                     ("log_lshare", "Labor share of output (REMPAG/VBP)"),
                     ("log_renimp", "Corporate income tax (RENIMP, profit proxy)")]:
        b, se, p, n = avgpost(c, col)
        rows.append({"outcome": lab, "beta": b, "se": se, "p": p, "n": n})
    did = pd.DataFrame(rows)
    did.to_csv(OUT / "incidence_did.csv", index=False)
    print("=== Incidence: avg post-2010 deviation (pre-only detrended, Post x MMI, MAIN) ===")
    print(did.round(4).to_string(index=False))
    print("\nReading: investment up while wage bill / labor share do not rise => the "
          "reconstruction's resources went to rebuilding capital, not to labor compensation; "
          "a non-negative RENIMP would corroborate recovering returns to capital.")


if __name__ == "__main__":
    main()
