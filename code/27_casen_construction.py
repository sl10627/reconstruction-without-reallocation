"""Upstream check (#2): is the manufacturing unskilled-wage rise plausibly driven by
reconstruction labor demand? The mechanism runs through the CONSTRUCTION sector, which
rebuilds and bids up the local low-skill wage. Here I show, in CASEN, that construction
employment and wages rose in harder-hit regions after the earthquake -- the upstream shock
behind the manufacturing wage-curve spillover.

CASEN identifies construction by industry (rama): CIIU Rev.2 (2006 wave) rama==5;
CIIU Rev.3 (2009+ waves) rama==6 (verified by ~8-9% employment share). Cells = region x
year (13-region scheme, merging Los Rios 14->10 and Arica 15->1, as in script 25).
Post = 1[year>=2010]. DiD with region + year FE, clustered by region.

Output: output/casen_construction_did.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "output"
MMI_FLOOR = 2.0
REGION_MERGE = {14: 10, 15: 1}


def mmi_13region():
    sh = pd.read_csv(DATA / "shakemap_by_region.csv").rename(columns={"enia_region": "REGION"})
    sh["mean_mmi"] = sh["mean_mmi"].fillna(MMI_FLOOR)
    sh["REGION13"] = sh["REGION"].replace(REGION_MERGE)
    return (sh.groupby("REGION13").apply(
        lambda g: (g["mean_mmi"] * g["region_area_m2"]).sum() / g["region_area_m2"].sum(),
        include_groups=False).rename("mean_mmi").reset_index().rename(columns={"REGION13": "REGION"}))


def build():
    d = pd.read_parquet(DATA / "casen_panel.parquet")
    d = d[d["activ"].isin([1, 2])].copy()                      # in labour force
    d["REGION"] = d["region15"].replace(REGION_MERGE)
    d["w"] = pd.to_numeric(d["expr"], errors="coerce")
    d["constr"] = ((d["rama_classif"] == "rev2") & (d["rama_raw"] == 5)) | \
                  ((d["rama_classif"] == "rev3") & (d["rama_raw"] == 6))
    d["wage"] = pd.to_numeric(d["wage_raw_real2017"], errors="coerce")

    rows = []
    for (reg, yr), g in d.groupby(["REGION", "year"]):
        c = g[g["constr"]]
        cw = c[(c["wage"] > 0)]
        rows.append({
            "REGION": int(reg), "year": int(yr),
            "constr_emp": c["w"].sum(),
            "total_emp": g["w"].sum(),
            "constr_share": c["w"].sum() / g["w"].sum(),
            "constr_wage": (cw["wage"] * cw["w"]).sum() / cw["w"].sum() if len(cw) else np.nan,
        })
    cell = pd.DataFrame(rows)
    cell = cell.merge(mmi_13region(), on="REGION", how="left")
    cell["mean_mmi"] = cell["mean_mmi"].fillna(MMI_FLOOR)
    cell["post"] = (cell["year"] >= 2010).astype(int)
    cell["post_mmi"] = cell["post"] * cell["mean_mmi"]
    cell["log_constr_emp"] = np.log(cell["constr_emp"])
    cell["log_constr_wage"] = np.log(cell["constr_wage"])
    return cell


def main():
    c = build()
    print(f"cells: {c.shape}; years={sorted(c.year.unique())}; regions={c.REGION.nunique()}")
    print("\nconstruction employment share by year (mean across regions):")
    print(c.groupby("year")["constr_share"].mean().round(3).to_dict())
    rows = []
    for col, lab in [("log_constr_emp", "Log construction employment"),
                     ("constr_share", "Construction emp. share"),
                     ("log_constr_wage", "Log construction wage")]:
        sub = c.dropna(subset=[col])
        m = smf.wls(f"{col} ~ post_mmi + C(REGION) + C(year)", data=sub,
                    weights=np.ones(len(sub))).fit(cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
        rows.append({"outcome": lab, "beta": m.params["post_mmi"], "se": m.bse["post_mmi"],
                     "p": m.pvalues["post_mmi"], "n": int(m.nobs)})
    did = pd.DataFrame(rows)
    did.to_csv(OUT / "casen_construction_did.csv", index=False)
    print("\n=== Construction sector, Post x MMI (CASEN, high-MMI vs low-MMI, post vs pre) ===")
    print(did.round(4).to_string(index=False))
    print("\nReading: positive coefficients => reconstruction raised local construction labor "
          "demand in harder-hit regions, the upstream shock behind the manufacturing wage spillover.")


if __name__ == "__main__":
    main()
