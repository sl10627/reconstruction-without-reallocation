"""Placebo / upstream check for the soil confound (see notes, Note 2).

Concern: shaking intensity (MMI) is correlated with soft-sediment alluvial soil, which is
also fertile. A sudden agricultural boom in fertile, high-MMI valleys around 2014 could load
onto food-related manufacturing and mimic the reconstruction investment pulse. This shock is
region x year, so it survives the industry x year fixed effects and linear detrending.

Test: does agriculture actually boom in harder-hit regions after the quake, peaking in 2014?
 (A) CASEN: agricultural employment share, wage, and log employment on Post x MMI, plus a
     wave event study (base 2009). Agriculture = rama_raw == 1 (ISIC major div. 1, rev2/rev3).
 (B) ENIA: food sector (ISIC 10) output and investment event studies, base 2009.

A genuine demand-driven agricultural boom would raise farm wages and food output and share the
pulse's 2014 peak-and-revert timing. Result: it does not. Farm wage and employment share do not
respond; farm employment drifts up monotonically through 2017 (wrong shape); food-sector output
shows no significant 2014 break even though food-sector investment shows the same 2014-2015
pulse as the aggregate (investment >> output = replacement, not expansion).

Output: output/agri_placebo.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "output"
MMI_FLOOR = 2.0
REGION_MERGE = {14: 10, 15: 1}  # CASEN 13-region scheme, as in scripts 25 and 27


def mmi_13region():
    sh = pd.read_csv(DATA / "shakemap_by_region.csv").rename(columns={"enia_region": "REGION"})
    sh["mean_mmi"] = sh["mean_mmi"].fillna(MMI_FLOOR)
    sh["REGION13"] = sh["REGION"].replace(REGION_MERGE)
    return (sh.groupby("REGION13").apply(
        lambda g: (g["mean_mmi"] * g["region_area_m2"]).sum() / g["region_area_m2"].sum(),
        include_groups=False).rename("mean_mmi").reset_index()
        .rename(columns={"REGION13": "REGION"}))


def casen_cells():
    d = pd.read_parquet(DATA / "casen_panel.parquet")
    d = d[d["activ"].isin([1, 2])].copy()                       # in labour force
    d["REGION"] = d["region15"].replace(REGION_MERGE)
    d["w"] = pd.to_numeric(d["expr"], errors="coerce")
    d["agri"] = (d["rama_raw"] == 1) & (d["rama_classif"].isin(["rev2", "rev3"]))
    d["wage"] = pd.to_numeric(d["wage_raw_real2017"], errors="coerce")
    rows = []
    for (reg, yr), g in d.groupby(["REGION", "year"]):
        a = g[g["agri"]]
        aw = a[a["wage"] > 0]
        rows.append({"REGION": int(reg), "year": int(yr),
                     "agri_emp": a["w"].sum(),
                     "agri_share": a["w"].sum() / g["w"].sum(),
                     "agri_wage": (aw["wage"] * aw["w"]).sum() / aw["w"].sum() if len(aw) else np.nan})
    c = pd.DataFrame(rows).merge(mmi_13region(), on="REGION", how="left", validate="m:1")
    c["mean_mmi"] = c["mean_mmi"].fillna(MMI_FLOOR)
    c["post"] = (c["year"] >= 2010).astype(int)
    c["post_mmi"] = c["post"] * c["mean_mmi"]
    c["log_agri_emp"] = np.log(c["agri_emp"])
    c["log_agri_wage"] = np.log(c["agri_wage"])
    return c


def did(cell, col, weights=None):
    sub = cell.dropna(subset=[col]).copy()
    w = np.ones(len(sub)) if weights is None else sub[weights].to_numpy()
    m = smf.wls(f"{col} ~ post_mmi + C(REGION) + C(year)", data=sub, weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
    return m.params["post_mmi"], m.bse["post_mmi"], m.pvalues["post_mmi"]


def event_study(cell, col, base=2009, weights=None):
    sub = cell.dropna(subset=[col]).copy()
    for k in sorted(sub.year.unique()):
        sub[f"y{k}"] = (sub.year == k).astype(int) * sub["mean_mmi"]
    terms = [f"y{k}" for k in sorted(sub.year.unique()) if k != base]
    w = np.ones(len(sub)) if weights is None else sub[weights].to_numpy()
    m = smf.wls(f"{col} ~ " + " + ".join(terms) + " + C(REGION) + C(year)",
                data=sub, weights=w).fit(cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
    return {int(t[1:]): (m.params[t], m.bse[t], m.pvalues[t]) for t in terms}


def main():
    out_rows = []

    # (A) CASEN agriculture
    c = casen_cells()
    print("CASEN national agri employment share by year:")
    print(c.groupby("year")["agri_share"].mean().round(3).to_dict())
    print("\n(A) CASEN DiD  Post x MMI  (region + year FE, cluster region)")
    for col, lab in [("agri_share", "Agri employment share"),
                     ("log_agri_wage", "Log agri wage"),
                     ("log_agri_emp", "Log agri employment")]:
        b, se, p = did(c, col)
        print(f"    {lab:24s} beta={b:+.4f}  se={se:.4f}  p={p:.3f}")
        out_rows.append({"source": "CASEN", "outcome": lab, "spec": "DiD", "year": np.nan,
                         "beta": b, "se": se, "p": p})
    print("\n(A) CASEN event study  Log agri employment  (base 2009)")
    for k, (b, se, p) in event_study(c, "log_agri_emp").items():
        print(f"    {k}: beta={b:+.4f}  se={se:.4f}  p={p:.3f}")
        out_rows.append({"source": "CASEN", "outcome": "Log agri employment",
                         "spec": "event", "year": k, "beta": b, "se": se, "p": p})

    # (B) ENIA food sector (ISIC 10)
    df = pd.read_parquet(DATA / "enia_cell_2digit_quake.parquet")
    df["year"] = df["ANIO"]
    food = df[df["ind2"] == "10"].copy()
    print("\n(B) ENIA food-sector (ISIC 10) event studies  (base 2009, weighted, cluster region)")
    for col, lab in [("log_vbp", "Food output (log VBP)"), ("log_inv", "Food investment (log)")]:
        print(f"  {lab}")
        for k, (b, se, p) in event_study(food, col, weights="weight_sum").items():
            flag = "  <--" if p < 0.10 else ""
            print(f"    {k}: beta={b:+.3f}  se={se:.3f}  p={p:.3f}{flag}")
            out_rows.append({"source": "ENIA", "outcome": lab, "spec": "event", "year": k,
                             "beta": b, "se": se, "p": p})

    OUT.mkdir(exist_ok=True)
    pd.DataFrame(out_rows).to_csv(OUT / "agri_placebo.csv", index=False)
    print(f"\nwrote {OUT / 'agri_placebo.csv'}")


if __name__ == "__main__":
    main()
