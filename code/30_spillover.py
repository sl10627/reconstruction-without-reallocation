"""Spillover / SUTVA diagnostic: are low-MMI control regions contaminated by
reconstruction reallocating materials and labor across regions?

Chile is effectively one-dimensional (a long, thin country), so geographic
adjacency is the north-south chain of regions. For each region I build a
"neighbor exposure" = the mean MMI of its immediate geographic neighbors, and
augment the main pre-period-projected DiD with a second treatment term:

  detr_{jrt} = beta (Post_t x MMI_own_r) + theta (Post_t x MMI_nbr_r)
               + C(jt) + e,   weights, clustered by region, MAIN sample.

This is a spatial-lag-of-treatment (SLX) test. Interpretation of theta:
  theta > 0 : low-MMI regions NEAR the core do BETTER after the quake (they
              supply inputs/labor and are pulled up) -> the high-vs-low contrast
              is ATTENUATED -> own-MMI beta is conservative (understates effect).
  theta < 0 : near-core controls do WORSE (competitive draining of resources to
              the core) -> contrast AMPLIFIED -> own-MMI beta overstated.
  theta ~ 0 : no detectable cross-region spillover; the relative DiD contrast is
              not obviously contaminated.

Caveat built in: with ~15 region clusters and two correlated treatment terms this
test is low-powered. The point is to replace a conjecture with whatever the data
actually say -- including "imprecise / inconclusive."

Output: output/spillover_test.csv
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

# Chile north -> south chain of ENIA region codes (one-dimensional geography).
# 15 Arica, 1 Tarapaca, 2 Antofagasta, 3 Atacama, 4 Coquimbo, 5 Valparaiso,
# 13 RM Santiago, 6 O'Higgins, 7 Maule, 8 Bio-Bio (incl. Nuble pre-2018),
# 9 Araucania, 14 Los Rios, 10 Los Lagos, 11 Aysen, 12 Magallanes.
NS_CHAIN = [15, 1, 2, 3, 4, 5, 13, 6, 7, 8, 9, 14, 10, 12]
# (11 Aysen has essentially no manufacturing cells; chain still defines its neighbors)
NS_CHAIN_FULL = [15, 1, 2, 3, 4, 5, 13, 6, 7, 8, 9, 14, 10, 11, 12]


def _sum(df, cols):
    cc = [c for c in cols if c in df.columns]
    return df[cc].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)


def neighbor_mmi(mmi_by_region: dict) -> dict:
    """Mean MMI of each region's immediate north/south neighbors in the chain."""
    chain = NS_CHAIN_FULL
    out = {}
    for i, r in enumerate(chain):
        nbrs = []
        if i > 0:
            nbrs.append(chain[i - 1])
        if i < len(chain) - 1:
            nbrs.append(chain[i + 1])
        vals = [mmi_by_region.get(n, MMI_FLOOR) for n in nbrs]
        out[r] = float(np.mean(vals)) if vals else MMI_FLOOR
    return out


def build():
    p = pd.read_parquet(DATA / "enia_pooled.parquet")
    p = p[p["ind_rev4"].notna() & p["REGION"].notna() & p["ANIO"].notna()].copy()
    p["ind2"] = p["ind_rev4"].astype(str).str[:2]
    p["REGION"] = pd.to_numeric(p["REGION"], errors="coerce")
    p["ANIO"] = pd.to_numeric(p["ANIO"], errors="coerce").astype(int)
    p["w"] = pd.to_numeric(p["ind_weight"], errors="coerce").fillna(0.0)
    p["vbp"] = pd.to_numeric(p["VBP"], errors="coerce")
    p["emp"] = pd.to_numeric(p["EMPTOT"], errors="coerce")
    p["inv"] = np.where(p["ANIO"] <= 2014, _sum(p, CBN), _sum(p, CB))

    def wsum(v, w):
        m = pd.notna(v) & (v > 0) & (w > 0)
        return (v[m] * w[m]).sum() if m.any() else np.nan

    rows = []
    for (i2, reg, yr), g in p.groupby(["ind2", "REGION", "ANIO"]):
        w = g["w"]
        rows.append({"ind2": i2, "REGION": int(reg), "ANIO": int(yr), "weight_sum": w.sum(),
                     "vbp": wsum(g["vbp"], w), "inv": wsum(g["inv"], w), "emp": wsum(g["emp"], w)})
    c = pd.DataFrame(rows)

    sh = pd.read_csv(DATA / "shakemap_by_region.csv").rename(columns={"enia_region": "REGION"})
    mmi_own = {int(r): (m if pd.notna(m) else MMI_FLOOR)
               for r, m in zip(sh["REGION"], sh["mean_mmi"])}
    mmi_nbr = neighbor_mmi(mmi_own)
    print("Region  own_MMI  neighbor_MMI")
    for r in NS_CHAIN_FULL:
        print(f"  {r:2d}     {mmi_own.get(r, MMI_FLOOR):.2f}     {mmi_nbr.get(r, MMI_FLOOR):.2f}")

    c["mean_mmi"] = c["REGION"].map(mmi_own).fillna(MMI_FLOOR)
    c["nbr_mmi"] = c["REGION"].map(mmi_nbr).fillna(MMI_FLOOR)
    c["year_c"] = c["ANIO"] - 2009
    c["jr"] = c["ind2"] + "_r" + c["REGION"].astype(str)
    c["jt"] = c["ind2"] + "_t" + c["ANIO"].astype(str)
    c["log_inv"] = np.log(c["inv"])
    c["log_vbp"] = np.log(c["vbp"])
    c["log_emp"] = np.where(c["emp"] > 0, np.log(c["emp"]), np.nan)
    return c


def spillover_did(c, col):
    """Pre-period-projected detrend (identical to incidence script), then DiD with
    both own and neighbor Post x MMI. Returns the augmented fit and the own-only fit."""
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
    post = (sub["ANIO"] >= 2010)
    sub["post_mmi"] = post * sub["mean_mmi"]
    sub["post_nbr"] = post * sub["nbr_mmi"]

    own = smf.wls("detr ~ post_mmi + C(jt)", data=sub, weights=sub[W].clip(lower=1)).fit(
        cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
    aug = smf.wls("detr ~ post_mmi + post_nbr + C(jt)", data=sub, weights=sub[W].clip(lower=1)).fit(
        cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
    return own, aug, int(aug.nobs)


def main():
    c = build()
    rows = []
    for col, lab in [("log_inv", "Investment"), ("log_vbp", "Gross output (VBP)"),
                     ("log_emp", "Employment")]:
        own, aug, n = spillover_did(c, col)
        rows.append({
            "outcome": lab, "n": n,
            "beta_own_only": own.params["post_mmi"], "se_own_only": own.bse["post_mmi"],
            "p_own_only": own.pvalues["post_mmi"],
            "beta_own_aug": aug.params["post_mmi"], "se_own_aug": aug.bse["post_mmi"],
            "p_own_aug": aug.pvalues["post_mmi"],
            "theta_nbr": aug.params["post_nbr"], "se_nbr": aug.bse["post_nbr"],
            "p_nbr": aug.pvalues["post_nbr"],
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "spillover_test.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== Spillover (SLX) test: own vs neighbor Post x MMI, MAIN sample ===")
    print(res.round(4).to_string(index=False))
    print("\nReading: theta_nbr is the neighbor-exposure coefficient. >0 => near-core controls "
          "pulled UP (contrast attenuated, own beta conservative); <0 => controls drained "
          "(contrast amplified); ~0 / insignificant => no detectable cross-region spillover. "
          "Compare beta_own_only vs beta_own_aug to see if controlling for neighbors moves the "
          "main estimate.")


if __name__ == "__main__":
    main()
