"""Gruber-style long-difference DiD.

Windows (mirrors Gruber's drop-transition approach):
  Clean pre:  2010, 2011, 2013
  Drop:       2008, 2009 (PRO/PRDI naming break + 2009 bytes-bug residual risk)
              2012 (low response rate)
              2014 (passage year, transition)
              2015 (first full phase-in year, transition)
  Clean post: 2016, 2017
  Wage post:  2015, 2016 (REGPRDI ends 2016; drop 2014 transition)

Capital intensity (treatment) = baseline (pre-period) CBU_total / EMPTOT, where
  CBU_total = sum of capital used by asset class: terrenos, edificios, maquinaria,
              vehiculos, muebles, software.

Spec: for each industry j, Delta_y_j = log(post_mean_y_j) - log(pre_mean_y_j).
Regress Delta_y_j on log(k_intensity_j) across industries.

Outputs:
  output/figs/B1_long_diff_scatter.png      6-panel scatter
  output/long_diff_table.csv                regression table
  output/long_diff_industry_data.csv        per-industry pre/post + treatment
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

PRE_YEARS = [2010, 2011, 2013]
POST_YEARS_MAIN = [2016, 2017]
POST_YEARS_WAGE = [2015, 2016]

CAP_USED_FIELDS = ["CBUTER", "CBUEDI", "CBUMAQ", "CBUVEH", "CBUMUE", "CBUSOF"]


def main():
    pool = pd.read_parquet(DATA / "enia_pooled.parquet")
    print(f"Pooled loaded: {pool.shape}")

    # build CBU_total = sum of all CBU asset classes (skipna)
    present = [c for c in CAP_USED_FIELDS if c in pool.columns]
    print(f"Capital fields present: {present}")
    cap = pool[present].apply(pd.to_numeric, errors="coerce")
    pool["CBU_total"] = cap.sum(axis=1, min_count=1)

    outcomes_main = ["EMPTOT", "CBU_total", "VBP", "RENIMP", "CONIMP", "TIMIMP"]
    outcomes_wage = ["REGPRDI", "REMHPRDI"]

    def industry_mean(years, outcome_cols):
        """Weighted mean per (industry, outcome) over the year subset."""
        sub = pool[pool["ANIO"].isin(years) & pool["ind_rev4"].notna()].copy()
        industries = sorted(sub["ind_rev4"].dropna().unique())
        result = pd.DataFrame(index=industries)
        result.index.name = "ind_rev4"
        for c in outcome_cols:
            if c not in sub.columns:
                result[c] = np.nan
                continue
            vals = pd.to_numeric(sub[c], errors="coerce")
            w = pd.to_numeric(sub["ind_weight"], errors="coerce")
            ok = vals.notna() & w.notna() & (w > 0)
            if ok.sum() == 0:
                result[c] = np.nan
                continue
            tmp = pd.DataFrame({
                "ind": sub["ind_rev4"][ok].values,
                "vw": (vals[ok] * w[ok]).values,
                "w":  w[ok].values,
            })
            agg = tmp.groupby("ind").agg(num=("vw", "sum"), den=("w", "sum"))
            agg["wmean"] = agg["num"] / agg["den"]
            result[c] = agg["wmean"]
        # effective N (weight sum) across all rows in the window for each industry
        wsum = (sub.groupby("ind_rev4")["ind_weight"].sum()).reindex(industries)
        result["n_eff"] = wsum
        return result

    pre = industry_mean(PRE_YEARS, outcomes_main + outcomes_wage)
    post_main = industry_mean(POST_YEARS_MAIN, outcomes_main)
    post_wage = industry_mean(POST_YEARS_WAGE, outcomes_wage)

    print(f"\nIndustries with any pre data: {len(pre)}")
    print(f"Industries with any post-main data: {len(post_main)}")
    print(f"Industries with any post-wage data: {len(post_wage)}")

    # Long differences: log(post) - log(pre), restrict to positive values
    df = pre.add_suffix("_pre").join(
        post_main.add_suffix("_post"), how="outer"
    ).join(
        post_wage.rename(columns={c: f"{c}_post" for c in outcomes_wage}), how="outer"
    )

    for c in outcomes_main:
        ppre = df[f"{c}_pre"]
        ppost = df[f"{c}_post"]
        valid = (ppre > 0) & (ppost > 0)
        df[f"d_log_{c}"] = np.where(valid, np.log(ppost) - np.log(ppre), np.nan)
    for c in outcomes_wage:
        ppre = df[f"{c}_pre"]
        ppost = df[f"{c}_post"]
        valid = (ppre > 0) & (ppost > 0)
        df[f"d_log_{c}"] = np.where(valid, np.log(ppost) - np.log(ppre), np.nan)

    # Capital intensity (treatment)
    df["k_KL"] = df["CBU_total_pre"] / df["EMPTOT_pre"]
    df["log_k_KL"] = np.where(df["k_KL"] > 0, np.log(df["k_KL"]), np.nan)
    df["k_KY"] = df["CBU_total_pre"] / df["VBP_pre"]
    df["labor_prod"] = df["VBP_pre"] / df["EMPTOT_pre"]
    df["log_labor_prod"] = np.where(df["labor_prod"] > 0, np.log(df["labor_prod"]), np.nan)

    df = df.reset_index().rename(columns={"index": "ind_rev4"})
    # Filter: require minimum effective N in both periods to avoid noisy industries
    df["n_eff_main"] = (df["n_eff_pre"].fillna(0) + df["n_eff_post"].fillna(0))
    df = df[df["n_eff_pre"].fillna(0) >= 10]  # at least 10 establishments in pre-period

    print(f"\nIndustries surviving filter (n_pre>=10): {len(df)}")
    df.to_csv(OUT / "long_diff_industry_data.csv", index=False)

    # Run long-difference regressions
    def lr(y_col, x_col, weight_col="n_eff_pre"):
        sub = df.dropna(subset=[y_col, x_col]).copy()
        if len(sub) < 5:
            return None, sub
        X = sm.add_constant(sub[x_col])
        if weight_col and weight_col in sub.columns:
            w = sub[weight_col].fillna(0).clip(lower=1)
            model = sm.WLS(sub[y_col], X, weights=w).fit()
        else:
            model = sm.OLS(sub[y_col], X).fit()
        return model, sub

    results = []
    outcomes_to_test = [
        ("d_log_EMPTOT",    "Employment"),
        ("d_log_VBP",       "Gross output (VBP)"),
        ("d_log_RENIMP",    "Corporate income tax"),
        ("d_log_CONIMP",    "Property/social tax"),
        ("d_log_TIMIMP",    "Other tax"),
        ("d_log_REGPRDI",   "Production wage (REGPRDI)"),
        ("d_log_REMHPRDI",  "Production hourly wage"),
        ("d_log_CBU_total", "Total capital (CBU)"),
    ]
    for ycol, name in outcomes_to_test:
        model, sub = lr(ycol, "log_k_KL")
        if model is None:
            print(f"  {name}: skipped (n={len(sub)})")
            continue
        b = model.params["log_k_KL"]
        se = model.bse["log_k_KL"]
        p = model.pvalues["log_k_KL"]
        n = int(model.nobs)
        r2 = model.rsquared
        mean_d = sub[ycol].mean()
        results.append({
            "outcome": name,
            "y_col": ycol,
            "beta": b,
            "se": se,
            "t_stat": b / se if se > 0 else np.nan,
            "p_value": p,
            "R2": r2,
            "n_industries": n,
            "mean_dy": mean_d,
        })
    tbl = pd.DataFrame(results)
    print("\n=== Long-difference DiD: beta on log(K/L) ===")
    print(tbl.round(3).to_string(index=False))
    tbl.to_csv(OUT / "long_diff_table.csv", index=False)

    # 6-panel scatter plot
    plot_outcomes = [
        ("d_log_EMPTOT",    "Δ log Employment",            "blue"),
        ("d_log_VBP",       "Δ log VBP (gross output)",    "darkgreen"),
        ("d_log_RENIMP",    "Δ log RENIMP (corp tax)",     "darkred"),
        ("d_log_REGPRDI",   "Δ log REGPRDI (prod wage)",   "purple"),
        ("d_log_CBU_total", "Δ log CBU (capital used)",    "darkorange"),
        ("d_log_CONIMP",    "Δ log CONIMP (placebo tax)",  "gray"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (ycol, ylab, color) in zip(axes.flat, plot_outcomes):
        sub = df.dropna(subset=[ycol, "log_k_KL"])
        if sub.empty:
            ax.set_title(f"{ylab}\n(no data)"); continue
        size = sub["n_eff_pre"].fillna(1).clip(lower=5)
        ax.scatter(sub["log_k_KL"], sub[ycol], alpha=0.45, s=size, color=color)
        # WLS fit line
        model, _ = lr(ycol, "log_k_KL")
        if model is not None:
            xx = np.linspace(sub["log_k_KL"].min(), sub["log_k_KL"].max(), 50)
            yy = model.params["const"] + model.params["log_k_KL"] * xx
            ax.plot(xx, yy, color="black", linewidth=1.5)
            b = model.params["log_k_KL"]
            se = model.bse["log_k_KL"]
            p = model.pvalues["log_k_KL"]
            ax.text(0.05, 0.95, f"β = {b:+.3f} ({se:.3f})\np = {p:.3f}, n = {int(model.nobs)}",
                    transform=ax.transAxes, va="top", fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray"))
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("log(K/L) baseline (CBU / EMPTOT, pre-period)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Long-difference DiD: industry-level Δ outcome vs baseline capital intensity\n"
        f"Pre: {PRE_YEARS}    Post (main): {POST_YEARS_MAIN}    Post (wage): {POST_YEARS_WAGE}    "
        f"Bubble size ∝ pre-period effective N",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "B1_long_diff_scatter.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {FIGS / 'B1_long_diff_scatter.png'}")
    print(f"Wrote {OUT / 'long_diff_table.csv'}")
    print(f"Wrote {OUT / 'long_diff_industry_data.csv'}")


if __name__ == "__main__":
    main()
