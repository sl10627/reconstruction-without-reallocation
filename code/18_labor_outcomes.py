"""Labor margin: does the reconstruction capital pulse reshape the skill structure
of manufacturing labor demand? (Capital-skill complementarity, Krusell et al. 2000.)

L1  Skill employment share = ESP / (ESP + production)
        skilled    = especialistas grid ESP{H,M}{1..4}T          (2003-2017)
        production = PRO grid (<=2008) bridged to PRDI grid (>=2009)
L2  Log skill premium = log( wage_TESP / wage_NOCALD )
        skilled wage    = REGTESP  (tecnicos/especialistas)       (2003-2016)
        unskilled wage  = REGNOCALD (no calificado)               (2003-2016)

Same design as the investment analysis: within-industry across-region DiD with
industry x region and industry x year FE; pre-period-projected detrending
(region trend from 2003-2009 only) for the event study.

Outputs:
  data/enia_cell_labor.parquet
  output/labor_did_main.csv             main Post x MMI (no trend / +region trend)
  output/labor_event_study.csv          pre-projected event study, L1 and L2
  output/figs/E7_labor_event_study.png
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
MMI_FLOOR = 2.0
W = "weight_sum"
PRE_YEARS = [2003, 2004, 2005, 2006, 2007, 2008, 2009]
ES_YEARS = [2003, 2004, 2005, 2006, 2007, 2008, 2010, 2011, 2013, 2014, 2015, 2016, 2017]  # 2009 ref

ESP = [f"ESP{g}{q}T" for g in "HM" for q in "1234"]
PRDI = [f"PRDI{g}{q}T" for g in "HM" for q in "1234"]
PRO = [f"PRO{g}{q}T" for g in "HM" for q in "1234"]

OUTCOMES = [("skill_share", "Skilled emp. share", False),     # level (share points)
            ("log_skillprem", "Log skill premium", False)]    # already a log ratio


def _sum(df, cols):
    cc = [c for c in cols if c in df.columns]
    return df[cc].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)


def build_cell():
    p = pd.read_parquet(DATA / "enia_pooled.parquet")
    p = p[p["ind_rev4"].notna() & p["REGION"].notna() & p["ANIO"].notna()].copy()
    p["ind2"] = p["ind_rev4"].astype(str).str[:2]
    p["REGION"] = pd.to_numeric(p["REGION"], errors="coerce")
    p["ANIO"] = pd.to_numeric(p["ANIO"], errors="coerce").astype(int)
    p["w"] = pd.to_numeric(p["ind_weight"], errors="coerce").fillna(0.0)

    p["esp_emp"] = _sum(p, ESP)
    prdi = _sum(p, PRDI)
    pro = _sum(p, PRO)
    p["prod_emp"] = np.where(p["ANIO"] <= 2008, pro, prdi)   # bridge PRO -> PRDI at 2009
    p["w_tesp"] = pd.to_numeric(p.get("REGTESP"), errors="coerce")
    p["w_nocald"] = pd.to_numeric(p.get("REGNOCALD"), errors="coerce")

    def wsum(v, w):
        m = v.notna() & (w > 0)
        return (v[m] * w[m]).sum() if m.any() else np.nan

    def wmean_pos(v, w):
        m = v.notna() & (v > 0) & (w > 0)
        return (v[m] * w[m]).sum() / w[m].sum() if m.any() else np.nan

    rows = []
    for (i2, reg, yr), g in p.groupby(["ind2", "REGION", "ANIO"]):
        w = g["w"]
        esp = wsum(g["esp_emp"], w)
        prod = wsum(g["prod_emp"], w)
        tesp = wmean_pos(g["w_tesp"], w)
        noc = wmean_pos(g["w_nocald"], w)
        rows.append({
            "ind2": i2, "REGION": int(reg), "ANIO": int(yr),
            "weight_sum": w.sum(),
            "skill_share": esp / (esp + prod) if (pd.notna(esp) and pd.notna(prod) and (esp + prod) > 0) else np.nan,
            "log_skillprem": np.log(tesp / noc) if (pd.notna(tesp) and pd.notna(noc) and tesp > 0 and noc > 0) else np.nan,
        })
    cell = pd.DataFrame(rows)

    shake = pd.read_csv(DATA / "shakemap_by_region.csv").rename(columns={"enia_region": "REGION"})
    cell = cell.merge(shake[["REGION", "mean_mmi"]], on="REGION", how="left", validate="m:1")
    cell["mean_mmi"] = cell["mean_mmi"].fillna(MMI_FLOOR)
    cell["post"] = (cell["ANIO"] >= 2010).astype(int)
    cell["year_c"] = cell["ANIO"] - 2009
    cell["high_mmi"] = (cell["mean_mmi"] >= 6.0).astype(int)
    cell["post_mmi"] = cell["post"] * cell["mean_mmi"]
    cell["jr"] = cell["ind2"] + "_r" + cell["REGION"].astype(str)
    cell["jt"] = cell["ind2"] + "_t" + cell["ANIO"].astype(str)
    cell.to_parquet(DATA / "enia_cell_labor.parquet", index=False)
    print(f"Wrote {DATA/'enia_cell_labor.parquet'}  {cell.shape}")
    return cell


def _fe_ok(sub, ycol):
    sub = sub.dropna(subset=[ycol]).copy()
    for k in ["jr", "jt"]:
        c = sub[k].value_counts()
        sub = sub[sub[k].isin(c[c >= 2].index)]
    return sub


def run_main(cell, trends=False):
    tt = " + C(REGION):year_c" if trends else ""
    rows = []
    for samp, sdf in [("MAIN", cell[cell["ind2"].isin(MAIN_IND2)]), ("ALL", cell)]:
        for col, lab, _ in OUTCOMES:
            sub = _fe_ok(sdf, col)
            if len(sub) < 25 or sub["REGION"].nunique() < 3:
                rows.append({"sample": samp, "outcome": lab, "trends": trends,
                             "beta": np.nan, "se": np.nan, "p": np.nan, "n": len(sub)})
                continue
            m = smf.wls(f"{col} ~ post_mmi + C(jr) + C(jt){tt}", data=sub,
                        weights=sub[W].clip(lower=1)).fit(cov_type="cluster",
                        cov_kwds={"groups": sub["REGION"]})
            rows.append({"sample": samp, "outcome": lab, "trends": trends,
                         "beta": m.params.get("post_mmi", np.nan), "se": m.bse.get("post_mmi", np.nan),
                         "p": m.pvalues.get("post_mmi", np.nan), "n": int(m.nobs)})
    return pd.DataFrame(rows)


def event_study_pretrend(cell):
    sdf = cell[cell["ind2"].isin(MAIN_IND2)].copy()
    rows = []
    for col, lab, _ in OUTCOMES:
        sub = _fe_ok(sdf, col)
        pre = sub[sub["ANIO"].isin(PRE_YEARS)]
        if pre.empty:
            continue
        jr_mean = (pre.groupby("jr").apply(lambda d: (d[col] * d[W]).sum() / d[W].sum(),
                                           include_groups=False).to_dict())
        sub = sub[sub["jr"].isin(jr_mean)].copy()
        sub["_dm"] = sub[col] - sub["jr"].map(jr_mean)
        slopes = {}
        for r, d in sub[sub["ANIO"].isin(PRE_YEARS)].groupby("REGION"):
            if d["year_c"].nunique() >= 2 and len(d) >= 3:
                mm = sm.WLS(d["_dm"], sm.add_constant(d["year_c"]), weights=d[W].clip(lower=1)).fit()
                slopes[r] = mm.params.get("year_c", 0.0)
            else:
                slopes[r] = 0.0
        sub["_detr"] = sub["_dm"] - sub["REGION"].map(slopes).fillna(0.0) * sub["year_c"]
        yrs = [y for y in ES_YEARS if (sub["ANIO"] == y).any()]
        for y in yrs:
            sub[f"m_{y}"] = sub["mean_mmi"] * (sub["ANIO"] == y).astype(int)
        terms = " + ".join(f"m_{y}" for y in yrs)
        m = smf.wls(f"_detr ~ {terms} + C(jt)", data=sub, weights=sub[W].clip(lower=1)).fit(
            cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
        for y in yrs:
            rows.append({"outcome": lab, "year": y, "beta": m.params.get(f"m_{y}", np.nan),
                         "se": m.bse.get(f"m_{y}", np.nan), "p": m.pvalues.get(f"m_{y}", np.nan)})
        rows.append({"outcome": lab, "year": 2009, "beta": 0.0, "se": 0.0, "p": np.nan})
    return pd.DataFrame(rows).sort_values(["outcome", "year"])


def plot_es(es):
    labs = es["outcome"].unique()
    fig, axes = plt.subplots(1, len(labs), figsize=(6 * len(labs), 4), squeeze=False)
    for ax, lab in zip(axes.flat, labs):
        d = es[(es["outcome"] == lab) & ~((es["se"] == 0) & (es["year"] != 2009))].sort_values("year")
        ax.errorbar(d["year"], d["beta"], yerr=1.96 * d["se"], marker="o", capsize=3, color="navy")
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(2009.5, color="darkred", ls="--", lw=1, label="earthquake")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("year"); ax.set_ylabel("beta (year x MMI)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Labor-demand response: skill share (L1) and skill premium (L2)\n"
                 "MAIN sample; pre-period-projected region trends; ref 2009", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "E7_labor_event_study.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {FIGS/'E7_labor_event_study.png'}")


def main():
    cell = build_cell()
    main_c = pd.concat([run_main(cell, False), run_main(cell, True)], ignore_index=True)
    main_c.to_csv(OUT / "labor_did_main.csv", index=False)
    print("\n=== L1/L2 main Post x MMI (no trend / +region trend) ===")
    print(main_c.round(4).to_string(index=False))

    es = event_study_pretrend(cell)
    es.to_csv(OUT / "labor_event_study.csv", index=False)
    print("\n=== L1/L2 event study (pre-projected) ===")
    print(es.round(4).to_string(index=False))
    plot_es(es)


if __name__ == "__main__":
    main()
