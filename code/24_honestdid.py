"""HonestDiD sensitivity analysis (Rambachan & Roth 2023) for the headline
event-study coefficients, replacing the ad-hoc linear detrending with a principled
robustness statement about how large a violation of parallel trends the conclusion
can tolerate.

Method. I feed the RAW (un-detrended) event study coefficients (year x MMI, ref
2009) and their cluster-robust covariance to HonestDiD, and report, for the 2014
peak coefficient:
  * the original CI (parallel trends imposed exactly);
  * Delta^SD(M): smoothness restriction. M=0 forces the differential trend to be
    exactly linear -- equivalent to my pre-period linear detrending; M>0 allows the
    trend to bend. (Caveat: the 2006/2012 gaps make the period spacing uneven, so
    Delta^SD treats observed periods as consecutive; read it as indicative.)
  * Delta^RM(Mbar): relative-magnitudes restriction. The post-period violation is at
    most Mbar times the largest pre-period violation. This is robust to uneven spacing
    and is the preferred restriction here.
I report the robust confidence sets and the breakdown value at which the effect first
becomes insignificant.

Targets: 2014 investment pulse; 2014 unskilled wage.
Output: output/honestdid_results.txt (printed), output/figs/E11_honestdid.png
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
MAIN_IND2 = ["10", "16", "24"]
MMI_FLOOR, W = 2.0, "weight_sum"
PRE = [2003, 2004, 2005, 2006, 2007, 2008]          # 2009 = reference (omitted)
POST = [2010, 2011, 2013, 2014, 2015, 2016, 2017]
YEARS = PRE + POST


def _sum(df, cols):
    cc = [c for c in cols if c in df.columns]
    return df[cc].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1) if cc else pd.Series(np.nan, index=df.index)


def build_pooled_cell(value_builder, how):
    p = pd.read_parquet(DATA / "enia_pooled.parquet")
    p = p[p["ind_rev4"].notna() & p["REGION"].notna() & p["ANIO"].notna()].copy()
    p["ind2"] = p["ind_rev4"].astype(str).str[:2]
    p["REGION"] = pd.to_numeric(p["REGION"], errors="coerce")
    p["ANIO"] = pd.to_numeric(p["ANIO"], errors="coerce").astype(int)
    p["w"] = pd.to_numeric(p["ind_weight"], errors="coerce").fillna(0.0)
    p["_v"] = value_builder(p)
    rows = []
    for (i2, reg, yr), g in p.groupby(["ind2", "REGION", "ANIO"]):
        v, w = g["_v"], g["w"]
        if how == "sum":
            m = pd.notna(v) & (w > 0); agg = (v[m] * w[m]).sum() if m.any() else np.nan
        else:
            m = v.notna() & (v > 0) & (w > 0); agg = (v[m] * w[m]).sum() / w[m].sum() if m.any() else np.nan
        rows.append({"ind2": i2, "REGION": int(reg), "ANIO": int(yr), "weight_sum": w.sum(), "val": agg})
    c = pd.DataFrame(rows)
    sh = pd.read_csv(DATA / "shakemap_by_region.csv").rename(columns={"enia_region": "REGION"})
    c = c.merge(sh[["REGION", "mean_mmi"]], on="REGION", how="left", validate="m:1")
    c["mean_mmi"] = c["mean_mmi"].fillna(MMI_FLOOR)
    c["jr"] = c["ind2"] + "_r" + c["REGION"].astype(str)
    c["jt"] = c["ind2"] + "_t" + c["ANIO"].astype(str)
    c["logval"] = np.where(c["val"] > 0, np.log(c["val"]), np.nan)
    return c


def raw_event_study(cell):
    """Return betahat (ordered PRE then POST) and cluster-robust sigma."""
    sub = cell[cell["ind2"].isin(MAIN_IND2)].dropna(subset=["logval"]).copy()
    for k in ["jr", "jt"]:
        vc = sub[k].value_counts(); sub = sub[sub[k].isin(vc[vc >= 2].index)]
    yrs = [y for y in YEARS if (sub["ANIO"] == y).any()]
    for y in yrs:
        sub[f"m_{y}"] = sub["mean_mmi"] * (sub["ANIO"] == y).astype(int)
    m = smf.wls(f"logval ~ {' + '.join(f'm_{y}' for y in yrs)} + C(jr) + C(jt)", data=sub,
                weights=sub[W].clip(lower=1)).fit(cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
    names = [f"m_{y}" for y in yrs]
    betahat = np.array([m.params[n] for n in names])
    V = m.cov_params()
    sigma = V.loc[names, names].values
    npre = sum(1 for y in yrs if y in PRE)
    npost = sum(1 for y in yrs if y in POST)
    post_years = [y for y in yrs if y in POST]
    return betahat, sigma, npre, npost, post_years


def make_lvec(post_years, target):
    """target='level' -> 2014 coefficient; target='spike' -> 2014 - (2013+2015)/2,
    a local second difference that nets out any (locally) smooth differential trend."""
    w = {y: 0.0 for y in post_years}
    if target == "level":
        w[2014] = 1.0
    else:  # spike contrast
        w[2014] = 1.0; w[2013] = -0.5; w[2015] = -0.5
    return np.array([[w[y]] for y in post_years])


def analyze(label, cell, target="level"):
    print(f"\n{'='*72}\n{label}   [target: {target}]\n{'='*72}")
    betahat, sigma, npre, npost, post_years = raw_event_study(cell)
    print(f"pre periods={npre}, post periods={npost}; post years={post_years}")
    l_vec = make_lvec(post_years, target)

    orig = hd.constructOriginalCS(betahat=betahat, sigma=sigma,
                                  numPrePeriods=npre, numPostPeriods=npost, l_vec=l_vec, alpha=0.05)
    olb, oub = float(orig["lb"].iloc[0]), float(orig["ub"].iloc[0])
    print(f"\nOriginal CI (parallel trends imposed) for 2014: [{olb:.3f}, {oub:.3f}]  "
          f"{'excludes 0' if (olb > 0 or oub < 0) else 'includes 0'}")

    def breakdown(df, key):
        for _, r in df.sort_values(key).iterrows():
            if r["lb"] <= 0 <= r["ub"]:
                return r[key]
        return None

    print("\nDelta^RM (relative magnitudes) robust 95% CIs for 2014:")
    rm = hd.createSensitivityResults_relativeMagnitudes(
        betahat=betahat, sigma=sigma, numPrePeriods=npre, numPostPeriods=npost,
        l_vec=l_vec, Mbarvec=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0], alpha=0.05)
    print(rm[["Mbar", "lb", "ub"]].round(3).to_string(index=False))
    bd_rm = breakdown(rm, "Mbar")
    print(f"  breakdown Mbar (CI first includes 0): {bd_rm}")

    print("\nDelta^SD (smoothness) robust 95% CIs for 2014:")
    sd = hd.createSensitivityResults(
        betahat=betahat, sigma=sigma, numPrePeriods=npre, numPostPeriods=npost,
        l_vec=l_vec, Mvec=[0.0, 0.02, 0.05, 0.1, 0.15, 0.2], alpha=0.05)
    print(sd[["M", "lb", "ub"]].round(3).to_string(index=False))
    bd_sd = breakdown(sd, "M")
    print(f"  breakdown M (CI first includes 0): {bd_sd}")

    return orig, rm, sd


def main():
    inv = build_pooled_cell(
        lambda p: np.where(p["ANIO"] <= 2014,
                           _sum(p, ["CBNTER", "CBNEDI", "CBNMAQ", "CBNVEH", "CBNMUE", "CBNSOF"]),
                           _sum(p, ["CBTER", "CBEDI", "CBMAQ", "CBVEH", "CBSOF"])), "sum")
    wage = build_pooled_cell(lambda p: pd.to_numeric(p.get("REGNOCALD"), errors="coerce"), "mean")

    # LEVEL estimand = the reported sensitivity (2014 coefficient)
    oL_inv, rmL_inv, sdL_inv = analyze("INVESTMENT (log)", inv, target="level")
    oL_wage, rmL_wage, sdL_wage = analyze("UNSKILLED WAGE (log REGNOCALD)", wage, target="level")
    # SPIKE contrast (reported in text as a negative result -- nets out the multi-year pulse)
    oS_inv, rmS_inv, sdS_inv = analyze("INVESTMENT (log)", inv, target="spike")
    oS_wage, rmS_wage, sdS_wage = analyze("UNSKILLED WAGE (log REGNOCALD)", wage, target="spike")

    def bd(df, key):
        for _, r in df.sort_values(key).iterrows():
            if r["lb"] <= 0 <= r["ub"]:
                return r[key]
        return float(df[key].max())

    # tidy CSV of the LEVEL-estimand results for the paper table
    rows = []
    for lab, o, rm, sd in [("Investment", oL_inv, rmL_inv, sdL_inv),
                           ("Unskilled wage", oL_wage, rmL_wage, sdL_wage)]:
        rows.append({"outcome": lab, "restriction": "Original (PT)", "param": np.nan,
                     "lb": float(o["lb"].iloc[0]), "ub": float(o["ub"].iloc[0])})
        for _, r in rm.iterrows():
            rows.append({"outcome": lab, "restriction": "RM", "param": r["Mbar"], "lb": r["lb"], "ub": r["ub"]})
        for _, r in sd.iterrows():
            rows.append({"outcome": lab, "restriction": "SD", "param": r["M"], "lb": r["lb"], "ub": r["ub"]})
        rows.append({"outcome": lab, "restriction": "breakdown_Mbar", "param": bd(rm, "Mbar"), "lb": np.nan, "ub": np.nan})
        rows.append({"outcome": lab, "restriction": "breakdown_M", "param": bd(sd, "M"), "lb": np.nan, "ub": np.nan})
        rows.append({"outcome": lab, "restriction": "spike_original", "param": np.nan,
                     "lb": float((oS_inv if lab == "Investment" else oS_wage)["lb"].iloc[0]),
                     "ub": float((oS_inv if lab == "Investment" else oS_wage)["ub"].iloc[0])})
    pd.DataFrame(rows).to_csv(OUT / "honestdid_results.csv", index=False)
    print(f"\nWrote {OUT/'honestdid_results.csv'}")

    # sensitivity plot (relative magnitudes), LEVEL estimand
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (lab, rm) in zip(axes, [("Investment", rmL_inv), ("Unskilled wage", rmL_wage)]):
        d = rm.sort_values("Mbar")
        ax.fill_between(d["Mbar"], d["lb"], d["ub"], alpha=0.25, color="steelblue",
                        label=r"$\Delta^{RM}(\bar M)$ robust 95% CI")
        ax.plot(d["Mbar"], d["lb"], color="steelblue"); ax.plot(d["Mbar"], d["ub"], color="steelblue")
        ax.axhline(0, color="darkred", ls="--", lw=1)
        ax.set_title(f"{lab}: 2014 effect sensitivity", fontsize=10)
        ax.set_xlabel(r"$\bar M$ (post / max pre violation)"); ax.set_ylabel("robust 95% CI for 2014 coef")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("HonestDiD sensitivity (Rambachan & Roth 2023): 2014 effect vs allowed parallel-trends violation",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(FIGS / "E11_honestdid.png", dpi=140); plt.close(fig)
    print(f"Wrote {FIGS/'E11_honestdid.png'}")


if __name__ == "__main__":
    main()
