"""Does the unskilled-wage rise reflect reconstruction labor DEMAND, or unskilled
out-migration (a labor-SUPPLY confound)? ENIA cannot tell them apart (it has no
skill-specific headcount past 2007). CASEN can: it is a household survey with region,
education (a clean skill proxy), and population expansion weights, so I can measure the
regional STOCK of unskilled vs skilled workers before and after the earthquake.

Logic. If unskilled workers LEFT harder-hit regions (supply story), the regional
unskilled population/labor force should FALL in high-MMI regions after 2010. If it does
not fall (the demand story), the supply confound is not operative.

CASEN waves: 2006, 2009 (pre) and 2011, 2013, 2015, 2017 (post). The earthquake is
Feb 27 2010; the CASEN 2009 wave is fielded in Nov 2009 (pre-quake) and there is no
2010 wave, so the first post-quake wave is 2011. I use the SAME convention as the ENIA
analysis: Post = 1[year >= 2010] and reference year 2009 (operationally Post selects
2011/13/15/17, since no 2010 wave exists). Skill = education: unskilled educc in {0,1,2}
(basic or less); skilled educc in {5,6} (tertiary). Cells are region x year, counts are
population-weighted (expr).

Spec: log(count)_{r,t} = beta (Post_t x MMI_r) + region FE + year FE, Post = 1[t>=2010],
clustered by region. Event study with 2009 as reference (mirrors the ENIA event study).

Output: output/casen_migration_did.csv, output/casen_migration_event_study.csv,
        output/figs/E12_casen_migration.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import honestdid as hd

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT, FIGS = ROOT / "data", ROOT / "output", ROOT / "output" / "figs"
MMI_FLOOR = 2.0
PRE = [2003, 2006, 2009]
POST = [2011, 2013, 2015, 2017]
ES_YEARS = [2003, 2006, 2011, 2013, 2015, 2017]  # 2009 = reference
# 13-region scheme (Los Rios 14 and Arica 15 split off only in 2007): merge to match 2003.
REGION_MERGE = {14: 10, 15: 1}


def mmi_13region():
    """Area-weighted MMI on the 13-region (pre-2007) scheme, merging 14->10, 15->1."""
    sh = pd.read_csv(DATA / "shakemap_by_region.csv").rename(columns={"enia_region": "REGION"})
    sh["mean_mmi"] = sh["mean_mmi"].fillna(MMI_FLOOR)
    sh["REGION13"] = sh["REGION"].replace(REGION_MERGE)
    out = (sh.groupby("REGION13")
             .apply(lambda g: (g["mean_mmi"] * g["region_area_m2"]).sum() / g["region_area_m2"].sum(),
                    include_groups=False)
             .rename("mean_mmi").reset_index().rename(columns={"REGION13": "REGION"}))
    return out


def _cells_from(df):
    """df has columns REGION, year, unskilled, skilled, in_lf, w."""
    rows = []
    for (reg, yr), g in df.groupby(["REGION", "year"]):
        rows.append({
            "REGION": int(reg), "year": int(yr),
            "pop_total": g["w"].sum(),
            "pop_unsk": g.loc[g["unskilled"], "w"].sum(),
            "pop_sk": g.loc[g["skilled"], "w"].sum(),
            "lf_unsk": g.loc[g["unskilled"] & g["in_lf"], "w"].sum(),
            "lf_sk": g.loc[g["skilled"] & g["in_lf"], "w"].sum(),
        })
    return pd.DataFrame(rows)


def build():
    # 2006-2017 panel, harmonized to 13 regions
    d = pd.read_parquet(DATA / "casen_panel.parquet")
    d = d[(d["edad"] >= 15) & (d["edad"] <= 64)].copy()
    d = d[d["educc"].notna() & (d["educc"] != 99)]
    d["REGION"] = d["region15"].replace(REGION_MERGE)
    d["unskilled"] = d["educc"].isin([0, 1, 2])
    d["skilled"] = d["educc"].isin([5, 6])
    d["in_lf"] = d["activ"].isin([1, 2])
    d["w"] = pd.to_numeric(d["expr"], errors="coerce")

    # 2003 (already working-age, 13 regions, harmonized in script 26)
    d03 = pd.read_parquet(DATA / "casen2003_slim.parquet")
    d03["year"] = 2003

    cells = pd.concat([_cells_from(d[["REGION", "year", "unskilled", "skilled", "in_lf", "w"]]),
                       _cells_from(d03)], ignore_index=True)
    cells["unsk_share"] = cells["pop_unsk"] / cells["pop_total"]
    cells = cells.merge(mmi_13region(), on="REGION", how="left", validate="m:1")
    cells["mean_mmi"] = cells["mean_mmi"].fillna(MMI_FLOOR)
    cells["post"] = (cells["year"] >= 2010).astype(int)   # same convention as ENIA
    cells["post_mmi"] = cells["post"] * cells["mean_mmi"]
    for k in ["pop_total", "pop_unsk", "pop_sk", "lf_unsk", "lf_sk"]:
        cells[f"log_{k}"] = np.log(cells[k])
    return cells


OUTCOMES = [("log_pop_unsk", "Unskilled population"),
            ("log_pop_sk", "Skilled population"),
            ("log_pop_total", "Total population"),
            ("log_lf_unsk", "Unskilled labor force"),
            ("unsk_share", "Unskilled share")]


def run_did(c):
    rows = []
    for col, lab in OUTCOMES:
        m = smf.wls(f"{col} ~ post_mmi + C(REGION) + C(year)", data=c,
                    weights=np.ones(len(c))).fit(cov_type="cluster", cov_kwds={"groups": c["REGION"]})
        rows.append({"outcome": lab, "beta": m.params["post_mmi"], "se": m.bse["post_mmi"],
                     "p": m.pvalues["post_mmi"], "n": int(m.nobs)})
    return pd.DataFrame(rows)


def event_study(c, col):
    cc = c.copy()
    for y in ES_YEARS:
        cc[f"m_{y}"] = cc["mean_mmi"] * (cc["year"] == y).astype(int)
    m = smf.wls(f"{col} ~ {' + '.join(f'm_{y}' for y in ES_YEARS)} + C(REGION) + C(year)",
                data=cc, weights=np.ones(len(cc))).fit(cov_type="cluster", cov_kwds={"groups": cc["REGION"]})
    rows = [{"year": y, "beta": m.params.get(f"m_{y}", np.nan), "se": m.bse.get(f"m_{y}", np.nan),
             "p": m.pvalues.get(f"m_{y}", np.nan)} for y in ES_YEARS]
    rows.append({"year": 2009, "beta": 0.0, "se": 0.0, "p": np.nan})
    return pd.DataFrame(rows).sort_values("year")


def pretrend_and_honest(c, col="log_pop_unsk"):
    """Formal pre-trend test + HonestDiD on the CASEN event study, mirroring the
    treatment of the investment/wage results. For a NULL (no out-migration) the
    relevant question is whether a LARGE out-migration effect (a big negative on the
    unskilled-population coefficient) can be ruled out under bounded trend violations."""
    cc = c.copy()
    yrs = [y for y in ES_YEARS if (cc["year"] == y).any()]          # 2003,2006 | 2011,13,15,17
    for y in yrs:
        cc[f"m_{y}"] = cc["mean_mmi"] * (cc["year"] == y).astype(int)
    m = smf.wls(f"{col} ~ {' + '.join(f'm_{y}' for y in yrs)} + C(REGION) + C(year)",
                data=cc, weights=np.ones(len(cc))).fit(
        cov_type="cluster", cov_kwds={"groups": cc["REGION"]})
    names = [f"m_{y}" for y in yrs]
    pre_names = [f"m_{y}" for y in yrs if y in PRE]
    # joint pre-trend Wald test (pre coefficients = 0)
    ftest = m.f_test(" = 0, ".join(pre_names) + " = 0")
    print(f"\nPre-trend joint test (CASEN, {col}): F={float(ftest.fvalue):.3f}, p={float(ftest.pvalue):.4f}")

    betahat = np.array([m.params[n] for n in names])
    sigma = m.cov_params().loc[names, names].values
    npre = sum(1 for y in yrs if y in PRE)
    npost = sum(1 for y in yrs if y in POST)
    post_years = [y for y in yrs if y in POST]
    l_avg = np.array([[1.0 / npost]] * npost)                       # average post effect
    orig = hd.constructOriginalCS(betahat=betahat, sigma=sigma, numPrePeriods=npre,
                                  numPostPeriods=npost, l_vec=l_avg, alpha=0.05)
    rm = hd.createSensitivityResults_relativeMagnitudes(
        betahat=betahat, sigma=sigma, numPrePeriods=npre, numPostPeriods=npost,
        l_vec=l_avg, Mbarvec=[0.5, 1.0, 2.0], alpha=0.05)
    print("HonestDiD on avg post unskilled-population effect (CASEN):")
    print(f"  Original CI: [{orig['lb'].iloc[0]:.3f}, {orig['ub'].iloc[0]:.3f}]")
    print(rm[["Mbar", "lb", "ub"]].round(3).to_string(index=False))
    # tidy save for the paper table
    out = [{"kind": "pretrend_F", "param": float(ftest.fvalue), "lb": np.nan, "ub": np.nan,
            "p": float(ftest.pvalue)},
           {"kind": "original", "param": np.nan, "lb": float(orig["lb"].iloc[0]),
            "ub": float(orig["ub"].iloc[0]), "p": np.nan}]
    for _, r in rm.iterrows():
        out.append({"kind": "RM", "param": r["Mbar"], "lb": r["lb"], "ub": r["ub"], "p": np.nan})
    pd.DataFrame(out).to_csv(OUT / "casen_honestdid.csv", index=False)
    return orig, rm


def main():
    c = build()
    print(f"CASEN region-year cells: {c.shape}; regions={c.REGION.nunique()}, years={sorted(c.year.unique())}")
    did = run_did(c)
    did.to_csv(OUT / "casen_migration_did.csv", index=False)
    print("\n=== Post x MMI on regional stocks (CASEN) ===")
    print(did.round(4).to_string(index=False))

    es_u = event_study(c, "log_pop_unsk"); es_u["outcome"] = "Unskilled population"
    es_s = event_study(c, "log_pop_sk"); es_s["outcome"] = "Skilled population"
    es = pd.concat([es_u, es_s], ignore_index=True)
    es.to_csv(OUT / "casen_migration_event_study.csv", index=False)
    print("\n=== Event study: unskilled population (key test) ===")
    print(es_u.round(4).to_string(index=False))

    print("\n=== Pre-trend test + HonestDiD (unskilled population) ===")
    pretrend_and_honest(c, "log_pop_unsk")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for lab, d, cl in [("Unskilled population", es_u, "darkorange"), ("Skilled population", es_s, "darkblue")]:
        ax.errorbar(d["year"], d["beta"], yerr=1.96 * d["se"], marker="o", capsize=3, label=lab, color=cl, alpha=0.85)
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(2010, color="darkred", ls="--", lw=1, label="earthquake")
    ax.set_xlabel("year"); ax.set_ylabel(r"$\beta$ (year $\times$ MMI)")
    ax.set_title("CASEN: regional population by skill vs earthquake exposure\n"
                 "(a fall in unskilled population would indicate out-migration / supply confound)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "E12_casen_migration.png", dpi=140); plt.close(fig)
    print(f"\nWrote {FIGS/'E12_casen_migration.png'}")


if __name__ == "__main__":
    main()
