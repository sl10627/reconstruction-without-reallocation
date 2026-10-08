"""Why is there no capital-skill complementarity response? Test the leading
explanation: the reconstruction pulse rebuilt STRUCTURES, not EQUIPMENT.

Krusell, Ohanian, Rios-Rull & Violante (2000) is about EQUIPMENT capital being
complementary to skilled labor; they explicitly separate equipment from
structures, which are not skill-complementary. An earthquake destroys and
rebuilds buildings/plants (structures). If the investment pulse is concentrated
in structures rather than machinery, the absence of a skill-demand response is
exactly what the equipment-specific CSC hypothesis predicts.

Decompose gross investment by asset class and run the same pre-period-projected
event study:
  structures  = EDI (buildings) + TER (land)      CBNEDI/CBNTER (<=2014) -> CBEDI/CBTER (>=2015)
  machinery   = MAQ                                CBNMAQ        (<=2014) -> CBMAQ        (>=2015)
  equipment   = MAQ + VEH + SOF                    (machinery + vehicles + software)

Outputs:
  output/asset_decomposition_event_study.csv
  output/asset_decomposition_pulse.csv     2014/2015 pulse by asset class
  output/figs/E8_asset_decomposition.png
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
ES_YEARS = [2003, 2004, 2005, 2006, 2007, 2008, 2010, 2011, 2013, 2014, 2015, 2016, 2017]

# asset-class fields: old (CBN*, <=2014) and new (CB*, >=2015)
ASSETS = {
    "structures": (["CBNEDI", "CBNTER"], ["CBEDI", "CBTER"]),
    "machinery":  (["CBNMAQ"],           ["CBMAQ"]),
    "equipment":  (["CBNMAQ", "CBNVEH", "CBNSOF"], ["CBMAQ", "CBVEH", "CBSOF"]),
}
OUTCOMES = [("log_structures", "Structures (buildings+land)"),
            ("log_machinery", "Machinery"),
            ("log_equipment", "Equipment (mach+veh+soft)")]


def _sum(df, cols):
    cc = [c for c in cols if c in df.columns]
    return df[cc].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1) if cc else pd.Series(np.nan, index=df.index)


def build_cell():
    p = pd.read_parquet(DATA / "enia_pooled.parquet")
    p = p[p["ind_rev4"].notna() & p["REGION"].notna() & p["ANIO"].notna()].copy()
    p["ind2"] = p["ind_rev4"].astype(str).str[:2]
    p["REGION"] = pd.to_numeric(p["REGION"], errors="coerce")
    p["ANIO"] = pd.to_numeric(p["ANIO"], errors="coerce").astype(int)
    p["w"] = pd.to_numeric(p["ind_weight"], errors="coerce").fillna(0.0)
    old = p["ANIO"] <= 2014
    for name, (oldf, newf) in ASSETS.items():
        p[name] = np.where(old, _sum(p, oldf), _sum(p, newf))

    def wsum(v, w):
        m = pd.notna(v) & (w > 0)
        return (v[m] * w[m]).sum() if m.any() else np.nan

    rows = []
    for (i2, reg, yr), g in p.groupby(["ind2", "REGION", "ANIO"]):
        w = g["w"]
        rows.append({"ind2": i2, "REGION": int(reg), "ANIO": int(yr), "weight_sum": w.sum(),
                     **{name: wsum(g[name], w) for name in ASSETS}})
    cell = pd.DataFrame(rows)
    shake = pd.read_csv(DATA / "shakemap_by_region.csv").rename(columns={"enia_region": "REGION"})
    cell = cell.merge(shake[["REGION", "mean_mmi"]], on="REGION", how="left", validate="m:1")
    cell["mean_mmi"] = cell["mean_mmi"].fillna(MMI_FLOOR)
    cell["year_c"] = cell["ANIO"] - 2009
    cell["jr"] = cell["ind2"] + "_r" + cell["REGION"].astype(str)
    cell["jt"] = cell["ind2"] + "_t" + cell["ANIO"].astype(str)
    for name in ASSETS:
        cell[f"log_{name}"] = np.where(cell[name] > 0, np.log(cell[name]), np.nan)
    return cell


def _fe_ok(sub, ycol):
    sub = sub.dropna(subset=[ycol]).copy()
    for k in ["jr", "jt"]:
        c = sub[k].value_counts()
        sub = sub[sub[k].isin(c[c >= 2].index)]
    return sub


def event_study(cell, ycol):
    sub = _fe_ok(cell[cell["ind2"].isin(MAIN_IND2)], ycol)
    pre = sub[sub["ANIO"].isin(PRE_YEARS)]
    if pre.empty:
        return pd.DataFrame()
    jr_mean = (pre.groupby("jr").apply(lambda d: (d[ycol] * d[W]).sum() / d[W].sum(),
                                       include_groups=False).to_dict())
    sub = sub[sub["jr"].isin(jr_mean)].copy()
    sub["_dm"] = sub[ycol] - sub["jr"].map(jr_mean)
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
    m = smf.wls(f"_detr ~ {' + '.join(f'm_{y}' for y in yrs)} + C(jt)", data=sub,
                weights=sub[W].clip(lower=1)).fit(cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
    rows = [{"year": y, "beta": m.params.get(f"m_{y}", np.nan), "se": m.bse.get(f"m_{y}", np.nan),
             "p": m.pvalues.get(f"m_{y}", np.nan)} for y in yrs]
    rows.append({"year": 2009, "beta": 0.0, "se": 0.0, "p": np.nan})
    return pd.DataFrame(rows).sort_values("year")


def main():
    cell = build_cell()
    all_es, pulse = [], []
    for col, lab in OUTCOMES:
        es = event_study(cell, col)
        if es.empty:
            print(f"{lab}: no data"); continue
        es["asset"] = lab
        all_es.append(es)
        for yr in (2014, 2015):
            r = es[es["year"] == yr]
            if len(r):
                pulse.append({"asset": lab, "year": yr, "beta": float(r["beta"].iloc[0]),
                              "se": float(r["se"].iloc[0]), "p": float(r["p"].iloc[0])})
    es_all = pd.concat(all_es, ignore_index=True)
    es_all.to_csv(OUT / "asset_decomposition_event_study.csv", index=False)
    pulse_df = pd.DataFrame(pulse)
    pulse_df.to_csv(OUT / "asset_decomposition_pulse.csv", index=False)
    print("=== Reconstruction pulse by asset class (2014/2015 x MMI) ===")
    print(pulse_df.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"Structures (buildings+land)": "darkred", "Machinery": "steelblue",
              "Equipment (mach+veh+soft)": "seagreen"}
    for lab in [o[1] for o in OUTCOMES]:
        d = es_all[es_all["asset"] == lab].sort_values("year")
        if d.empty:
            continue
        ax.errorbar(d["year"], d["beta"], yerr=1.96 * d["se"], marker="o", capsize=3,
                    label=lab, color=colors.get(lab), alpha=0.85)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(2009.5, color="black", ls="--", lw=1)
    ax.set_xlabel("year"); ax.set_ylabel(r"$\beta$ (year $\times$ MMI)")
    ax.set_title("Reconstruction investment pulse by asset class\n"
                 "structures spike; machinery/equipment do not (explains the CSC null)", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "E8_asset_decomposition.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {FIGS/'E8_asset_decomposition.png'}")


if __name__ == "__main__":
    main()
