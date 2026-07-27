"""INFEASIBLE (kept as a diagnostic record). The ENIA hours variables change
DEFINITION, not just name, at the 2011 redesign: pre-2011 HRS* are annual hours
(~2100/worker) while post-2011 HRSE*+HREX* are on a ~25x smaller scale (~80),
so hours-per-worker is not comparable across the 2010 treatment window. No
total-hours field (HDIA/DDIA/TEED) is a clean annual-hours measure either.
=> The intensive (hours) margin cannot be measured across the earthquake; the
paper reports this as a data limitation and uses only the extensive margin
(employment). Original docstring follows.

L3: intensive vs extensive labor margin. Did establishments in harder-hit
regions adjust labor through more workers (extensive, = employment, already shown
flat/transient) or more hours per worker (intensive)?

Intensive margin = total hours / total workers. Total occupational hours were
renamed at the 2011 form redesign (like the investment fields):
  <=2010: HRS* by occupation (total hours)         e.g. HRSESPEC, HRSNOCALD, HRSPRDI, ...
  >=2011: HRSE* (standard) + HREX* (overtime) by occupation
I unify them into total hours; the 2011 redefinition is common within industry-year
and is absorbed by the industry x year fixed effects (same logic as CBN->CB).

hours_per_worker = (cell total hours) / (cell total EMPTOT). Pre-period-projected
event study + main DiD, MAIN sample.

Output: output/hours_margin_did.csv, output/hours_margin_event_study.csv,
        output/figs/E10_hours_margin.png
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
PRE_YEARS = [2003, 2004, 2005, 2007, 2008, 2009]
ES_YEARS = [2003, 2004, 2005, 2007, 2008, 2010, 2011, 2013, 2014, 2015, 2016, 2017]

OCC = ["ESPEC", "ADMIN", "NOCALD", "COMS", "AUX", "SERV", "DIREC", "PROP", "PRDI", "ADSE"]
HRS_OLD = [f"HRS{o}" for o in OCC]                       # total hours by occupation, <=2010
OCC_NEW = ["PRDI", "ESP", "ADSE", "COMS", "AUX", "NOCALD"]
HRSE_NEW = [f"HRSE{o}" for o in OCC_NEW] + [f"HREX{o}" for o in OCC_NEW]  # standard + overtime, >=2011


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
    p["tothours"] = np.where(p["ANIO"] <= 2010, _sum(p, HRS_OLD), _sum(p, HRSE_NEW))
    p["emp"] = pd.to_numeric(p["EMPTOT"], errors="coerce")

    def wsum(v, w):
        m = pd.notna(v) & (w > 0); return (v[m] * w[m]).sum() if m.any() else np.nan

    rows = []
    for (i2, reg, yr), g in p.groupby(["ind2", "REGION", "ANIO"]):
        h, e = wsum(g["tothours"], g["w"]), wsum(g["emp"], g["w"])
        rows.append({"ind2": i2, "REGION": int(reg), "ANIO": int(yr), "weight_sum": g["w"].sum(),
                     "hpw": h / e if (pd.notna(h) and pd.notna(e) and e > 0) else np.nan})
    c = pd.DataFrame(rows)
    sh = pd.read_csv(DATA / "shakemap_by_region.csv").rename(columns={"enia_region": "REGION"})
    c = c.merge(sh[["REGION", "mean_mmi"]], on="REGION", how="left")
    c["mean_mmi"] = c["mean_mmi"].fillna(MMI_FLOOR)
    c["year_c"] = c["ANIO"] - 2009
    c["post_mmi"] = (c["ANIO"] >= 2010) * c["mean_mmi"]
    c["jr"] = c["ind2"] + "_r" + c["REGION"].astype(str)
    c["jt"] = c["ind2"] + "_t" + c["ANIO"].astype(str)
    c["log_hpw"] = np.where(c["hpw"] > 0, np.log(c["hpw"]), np.nan)
    return c


def _fe_ok(sub, ycol):
    sub = sub.dropna(subset=[ycol]).copy()
    for k in ["jr", "jt"]:
        vc = sub[k].value_counts(); sub = sub[sub[k].isin(vc[vc >= 2].index)]
    return sub


def detrend(cell, ycol):
    sub = _fe_ok(cell[cell["ind2"].isin(MAIN_IND2)], ycol)
    pre = sub[sub["ANIO"].isin(PRE_YEARS)]
    jr_mean = pre.groupby("jr").apply(lambda d: (d[ycol] * d[W]).sum() / d[W].sum(),
                                      include_groups=False).to_dict()
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
    print("hours-per-worker by year (cell median, sanity check):")
    print(cell.groupby("ANIO")["hpw"].median().round(0).to_dict())

    # main DiD (no trend / + region trend)
    rows = []
    for trends in (False, True):
        tt = " + C(REGION):year_c" if trends else ""
        sub = _fe_ok(cell[cell["ind2"].isin(MAIN_IND2)], "log_hpw")
        m = smf.wls(f"log_hpw ~ post_mmi + C(jr) + C(jt){tt}", data=sub,
                    weights=sub[W].clip(lower=1)).fit(cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
        rows.append({"trends": trends, "beta": m.params["post_mmi"], "se": m.bse["post_mmi"],
                     "p": m.pvalues["post_mmi"], "n": int(m.nobs)})
    did = pd.DataFrame(rows)
    did.to_csv(OUT / "hours_margin_did.csv", index=False)
    print("\n=== Intensive margin (log hours/worker), Post x MMI ===")
    print(did.round(4).to_string(index=False))

    # event study (pre-projected)
    sub = detrend(cell, "log_hpw")
    yrs = [y for y in ES_YEARS if (sub["ANIO"] == y).any()]
    for y in yrs:
        sub[f"m_{y}"] = sub["mean_mmi"] * (sub["ANIO"] == y)
    m = smf.wls(f"_detr ~ {' + '.join(f'm_{y}' for y in yrs)} + C(jt)", data=sub,
                weights=sub[W].clip(lower=1)).fit(cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
    es = [{"year": y, "beta": m.params.get(f"m_{y}", np.nan), "se": m.bse.get(f"m_{y}", np.nan),
           "p": m.pvalues.get(f"m_{y}", np.nan)} for y in yrs]
    es.append({"year": 2009, "beta": 0.0, "se": 0.0, "p": np.nan})
    es = pd.DataFrame(es).sort_values("year")
    es.to_csv(OUT / "hours_margin_event_study.csv", index=False)
    print("\n=== Event study (log hours/worker), key years ===")
    print(es[es["year"].isin([2008, 2009, 2010, 2014, 2016])].round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(es["year"], es["beta"], yerr=1.96 * es["se"], marker="o", capsize=3, color="teal")
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(2009.5, color="darkred", ls="--", lw=1)
    ax.set_xlabel("year"); ax.set_ylabel(r"$\beta$ (year $\times$ MMI)")
    ax.set_title("Intensive margin: log hours per worker\n(pre-projected; 2011 hours redefinition absorbed by FE)", fontsize=10)
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(FIGS / "E10_hours_margin.png", dpi=140); plt.close(fig)
    print(f"\nWrote {FIGS/'E10_hours_margin.png'}")


if __name__ == "__main__":
    main()
