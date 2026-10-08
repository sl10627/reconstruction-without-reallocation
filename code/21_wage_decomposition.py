"""Decompose the skill-premium decline: is it the skilled wage (REGTESP) falling,
or the unskilled wage (REGNOCALD) rising? If the unskilled wage rises in high-MMI
regions, that pins the mechanism: reconstruction raises demand for less-skilled /
manual labor, compressing the premium and swamping any capital-skill complementarity.

Both wages available 2003-2016. Cell = weighted mean of each occupational monthly
wage. Pre-period-projected detrending (region trend on 2003-2009 only), MAIN sample.

Output: output/wage_decomposition_event_study.csv ; output/wage_decomposition_summary.csv
        output/figs/E9_wage_decomposition.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT, FIGS = ROOT / "data", ROOT / "output", ROOT / "output" / "figs"
MAIN_IND2 = ["10", "16", "24"]
MMI_FLOOR, W = 2.0, "weight_sum"
PRE_YEARS = [2003, 2004, 2005, 2006, 2007, 2008, 2009]
ES_YEARS = [2003, 2004, 2005, 2006, 2007, 2008, 2010, 2011, 2013, 2014, 2015, 2016]  # 2009 ref; no 2017
OUTCOMES = [("log_wskill", "Skilled wage (REGTESP)"), ("log_wunsk", "Unskilled wage (REGNOCALD)")]


def build():
    p = pd.read_parquet(DATA / "enia_pooled.parquet")
    p = p[p["ind_rev4"].notna() & p["REGION"].notna() & p["ANIO"].notna()].copy()
    p["ind2"] = p["ind_rev4"].astype(str).str[:2]
    p["REGION"] = pd.to_numeric(p["REGION"], errors="coerce")
    p["ANIO"] = pd.to_numeric(p["ANIO"], errors="coerce").astype(int)
    p["w"] = pd.to_numeric(p["ind_weight"], errors="coerce").fillna(0.0)
    p["wskill"] = pd.to_numeric(p.get("REGTESP"), errors="coerce")
    p["wunsk"] = pd.to_numeric(p.get("REGNOCALD"), errors="coerce")

    def wmean_pos(v, w):
        m = v.notna() & (v > 0) & (w > 0)
        return (v[m] * w[m]).sum() / w[m].sum() if m.any() else np.nan

    rows = []
    for (i2, reg, yr), g in p.groupby(["ind2", "REGION", "ANIO"]):
        rows.append({"ind2": i2, "REGION": int(reg), "ANIO": int(yr), "weight_sum": g["w"].sum(),
                     "wskill": wmean_pos(g["wskill"], g["w"]), "wunsk": wmean_pos(g["wunsk"], g["w"])})
    c = pd.DataFrame(rows)
    sh = pd.read_csv(DATA / "shakemap_by_region.csv").rename(columns={"enia_region": "REGION"})
    c = c.merge(sh[["REGION", "mean_mmi"]], on="REGION", how="left", validate="m:1")
    c["mean_mmi"] = c["mean_mmi"].fillna(MMI_FLOOR)
    c["year_c"] = c["ANIO"] - 2009
    c["jr"] = c["ind2"] + "_r" + c["REGION"].astype(str)
    c["jt"] = c["ind2"] + "_t" + c["ANIO"].astype(str)
    c["log_wskill"] = np.where(c["wskill"] > 0, np.log(c["wskill"]), np.nan)
    c["log_wunsk"] = np.where(c["wunsk"] > 0, np.log(c["wunsk"]), np.nan)
    return c


def detrend(cell, ycol):
    sub = cell[cell["ind2"].isin(MAIN_IND2)].dropna(subset=[ycol]).copy()
    for k in ["jr", "jt"]:
        vc = sub[k].value_counts(); sub = sub[sub[k].isin(vc[vc >= 2].index)]
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
    es_rows, summ = [], []
    for col, lab in OUTCOMES:
        sub = detrend(cell, col)
        # event study
        yrs = [y for y in ES_YEARS if (sub["ANIO"] == y).any()]
        for y in yrs:
            sub[f"m_{y}"] = sub["mean_mmi"] * (sub["ANIO"] == y).astype(int)
        m = smf.wls(f"_detr ~ {' + '.join(f'm_{y}' for y in yrs)} + C(jt)", data=sub,
                    weights=sub[W].clip(lower=1)).fit(cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
        for y in yrs:
            es_rows.append({"outcome": lab, "year": y, "beta": m.params.get(f"m_{y}", np.nan),
                            "se": m.bse.get(f"m_{y}", np.nan), "p": m.pvalues.get(f"m_{y}", np.nan)})
        es_rows.append({"outcome": lab, "year": 2009, "beta": 0.0, "se": 0.0, "p": np.nan})
        # average post + early/late
        sub["post_mmi"] = (sub["ANIO"] >= 2010) * sub["mean_mmi"]
        sub["early_mmi"] = sub["ANIO"].between(2010, 2014) * sub["mean_mmi"]
        sub["late_mmi"] = (sub["ANIO"] >= 2015) * sub["mean_mmi"]
        a = smf.wls("_detr ~ post_mmi + C(jt)", data=sub, weights=sub[W].clip(lower=1)).fit(
            cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
        b = smf.wls("_detr ~ early_mmi + late_mmi + C(jt)", data=sub, weights=sub[W].clip(lower=1)).fit(
            cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
        summ.append({"wage": lab, "avg_post": a.params["post_mmi"], "p": a.pvalues["post_mmi"],
                     "early_2010_14": b.params["early_mmi"], "early_p": b.pvalues["early_mmi"],
                     "late_2015_16": b.params["late_mmi"], "late_p": b.pvalues["late_mmi"]})
    es = pd.DataFrame(es_rows).sort_values(["outcome", "year"])
    es.to_csv(OUT / "wage_decomposition_event_study.csv", index=False)
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "wage_decomposition_summary.csv", index=False)
    print("=== Wage decomposition: avg post-2010 deviation from pre-trend (per MMI) ===")
    print(s.round(4).to_string(index=False))
    print("\n=== Event study (key years) ===")
    print(es[es["year"].isin([2008, 2009, 2010, 2014, 2016])].round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lab, cl in [("Skilled wage (REGTESP)", "darkblue"), ("Unskilled wage (REGNOCALD)", "darkorange")]:
        d = es[es["outcome"] == lab].sort_values("year")
        ax.errorbar(d["year"], d["beta"], yerr=1.96 * d["se"], marker="o", capsize=3, label=lab, color=cl, alpha=0.85)
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(2009.5, color="black", ls="--", lw=1)
    ax.set_xlabel("year"); ax.set_ylabel(r"$\beta$ (year $\times$ MMI)")
    ax.set_title("Decomposing the skill-premium decline:\nskilled vs unskilled wage response to earthquake exposure", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "E9_wage_decomposition.png", dpi=140); plt.close(fig)
    print(f"\nWrote {FIGS/'E9_wage_decomposition.png'}")


if __name__ == "__main__":
    main()
