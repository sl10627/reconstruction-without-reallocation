"""Manufacturing aggregate trends + productivity DiD.

Closes the gap between channel-level analysis and aggregate manufacturing response.

Part 1: Aggregate descriptive trends 2008-2017
  Total VBP, Total VA, Total EMPTOT, N_establishments, labor productivity.
  These are sample-level (ENIA universe of 10+ worker manufacturing firms).

Part 2: Productivity long-difference DiD
  Outcome: Δ log(VA / EMPTOT) — labor productivity
  Outcome: Δ log(VBP / EMPTOT) — labor productivity gross
  Treatment: baseline log(investment intensity) (from B3 main spec)
  Same robustness framework: pre-trend + placebo
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

OLD_INV = ["CBNEDI", "CBNMAQ", "CBNVEH", "CBNSOF", "CBNTER", "CBNMUE", "CBNOAT", "CBNOTRI"]
NEW_INV = ["CBEDI", "CBMAQ", "CBVEH", "CBSOF", "CBTER", "CBOAT", "CBOTRI"]


def add_capital(pool):
    old = pool["ANIO"] <= 2014
    new = pool["ANIO"] >= 2015
    def s(df, cols):
        present = [c for c in cols if c in df.columns]
        return df[present].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    pool["cap_inv_total"] = np.nan
    pool.loc[old, "cap_inv_total"] = s(pool.loc[old], OLD_INV)
    pool.loc[new, "cap_inv_total"] = s(pool.loc[new], NEW_INV)
    return pool


def main():
    pool = pd.read_parquet(DATA / "enia_pooled.parquet")
    pool = add_capital(pool)

    # ============================================================
    # Part 1: Aggregate manufacturing trends 2008-2017
    # ============================================================
    # Per row, compute outcome * weight, then sum per year for "total"
    def weighted_year_total(df, col):
        v = pd.to_numeric(df[col], errors="coerce")
        w = pd.to_numeric(df["ind_weight"], errors="coerce")
        ok = v.notna() & w.notna()
        return (df.loc[ok].assign(vw=v[ok] * w[ok])
                  .groupby("ANIO")["vw"].sum())

    def weighted_year_mean(df, col):
        v = pd.to_numeric(df[col], errors="coerce")
        w = pd.to_numeric(df["ind_weight"], errors="coerce")
        ok = v.notna() & w.notna() & (w > 0)
        return (df.loc[ok].assign(vw=v[ok] * w[ok], w=w[ok])
                  .groupby("ANIO")
                  .apply(lambda g: g["vw"].sum() / g["w"].sum(), include_groups=False))

    agg = pd.DataFrame({
        "VBP_total":  weighted_year_total(pool, "VBP"),
        "VA_total":   weighted_year_total(pool, "VA"),
        "EMPTOT_total": weighted_year_total(pool, "EMPTOT"),
        "cap_inv_total_sum":  weighted_year_total(pool, "cap_inv_total"),
        "RENIMP_total": weighted_year_total(pool, "RENIMP"),
        "N_est":      pool.groupby("ANIO")["ind_weight"].sum(),
        "VA_mean":    weighted_year_mean(pool, "VA"),
        "VBP_mean":   weighted_year_mean(pool, "VBP"),
        "EMPTOT_mean": weighted_year_mean(pool, "EMPTOT"),
    })
    # labor productivity
    agg["labor_prod_VA"]  = agg["VA_total"]  / agg["EMPTOT_total"]
    agg["labor_prod_VBP"] = agg["VBP_total"] / agg["EMPTOT_total"]

    print("=== Aggregate manufacturing trends (ENIA-universe) ===")
    print(agg.round(0).to_string())
    agg.to_csv(OUT / "manufacturing_aggregate.csv")

    # ----- Plot: 6-panel aggregate trends -----
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        ("VBP_total",      "Total gross output (VBP)",     "CLP nominal (sum)",       "darkgreen"),
        ("VA_total",       "Total value added (VA)",       "CLP nominal (sum)",       "tab:olive"),
        ("EMPTOT_total",   "Total employment",             "headcount (sum)",         "tab:blue"),
        ("N_est",          "Number of establishments",     "count",                   "tab:purple"),
        ("labor_prod_VA",  "Labor productivity (VA/L)",    "CLP per worker",          "tab:orange"),
        ("RENIMP_total",   "Total corporate income tax",   "CLP nominal (sum)",       "tab:red"),
    ]
    for ax, (col, title, ylab, color) in zip(axes.flat, panels):
        ax.plot(agg.index, agg[col], marker="o", color=color, linewidth=2)
        ax.axvline(2014, color="red", linestyle="--", alpha=0.5, label="2014 reform")
        ax.set_xlabel("Year"); ax.set_ylabel(ylab)
        ax.set_title(title); ax.legend(loc="best"); ax.grid(alpha=0.3)
    fig.suptitle("ENIA-universe Chile manufacturing aggregates 2008-2017 "
                 "(weighted by share-weighted industry-year)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "D1_manufacturing_aggregates.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {FIGS / 'D1_manufacturing_aggregates.png'}")

    # ----- Plot: indexed to 2013=100 -----
    idx = agg.div(agg.loc[2013]) * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    for col, color, label in [
        ("VBP_total",      "darkgreen", "Gross output"),
        ("VA_total",       "tab:olive", "Value added"),
        ("EMPTOT_total",   "tab:blue",  "Employment"),
        ("N_est",          "tab:purple","N establishments"),
        ("labor_prod_VA",  "tab:orange","Labor productivity (VA/L)"),
        ("RENIMP_total",   "tab:red",   "Corporate tax revenue"),
    ]:
        ax.plot(idx.index, idx[col], marker="o", color=color, label=label, linewidth=2)
    ax.axvline(2014, color="black", linestyle="--", alpha=0.5, label="2014 reform")
    ax.axhline(100, color="gray", linewidth=0.5)
    ax.set_xlabel("Year"); ax.set_ylabel("Index, 2013 = 100")
    ax.set_title("Chile manufacturing aggregate trends, 2013 = 100")
    ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "D2_manufacturing_indexed.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {FIGS / 'D2_manufacturing_indexed.png'}")

    # ============================================================
    # Part 2: Productivity long-difference DiD
    # ============================================================
    # Build industry-year labor productivity, then long-diff
    PRE_YEARS  = [2010, 2011, 2013]
    POST_YEARS = [2016, 2017]

    def industry_mean(years, outcome_cols):
        sub = pool[pool["ANIO"].isin(years) & pool["ind_rev4"].notna()].copy()
        industries = sorted(sub["ind_rev4"].dropna().unique())
        result = pd.DataFrame(index=industries)
        result.index.name = "ind_rev4"
        for c in outcome_cols:
            if c not in sub.columns:
                result[c] = np.nan; continue
            vals = pd.to_numeric(sub[c], errors="coerce")
            w = pd.to_numeric(sub["ind_weight"], errors="coerce")
            ok = vals.notna() & w.notna() & (w > 0)
            if ok.sum() == 0:
                result[c] = np.nan; continue
            tmp = pd.DataFrame({"ind": sub["ind_rev4"][ok].values,
                                "vw": (vals[ok] * w[ok]).values,
                                "w":  w[ok].values})
            agg = tmp.groupby("ind").agg(num=("vw","sum"), den=("w","sum"))
            result[c] = agg["num"] / agg["den"]
        result["n_eff"] = (sub.groupby("ind_rev4")["ind_weight"].sum()).reindex(industries)
        return result

    outcomes = ["EMPTOT", "VBP", "VA", "cap_inv_total"]
    pre  = industry_mean(PRE_YEARS,  outcomes)
    post = industry_mean(POST_YEARS, outcomes)

    df = pre.add_suffix("_pre").join(post.add_suffix("_post"), how="outer")
    # Build labor productivity outcome (VA / EMPTOT)
    df["labor_prod_pre"]  = df["VA_pre"]  / df["EMPTOT_pre"]
    df["labor_prod_post"] = df["VA_post"] / df["EMPTOT_post"]
    df["labor_prod_VBP_pre"]  = df["VBP_pre"]  / df["EMPTOT_pre"]
    df["labor_prod_VBP_post"] = df["VBP_post"] / df["EMPTOT_post"]
    # Long differences
    for c in ["labor_prod", "labor_prod_VBP", "VBP", "VA"]:
        a, b = df[f"{c}_pre"], df[f"{c}_post"]
        valid = (a > 0) & (b > 0)
        df[f"d_log_{c}"] = np.where(valid, np.log(b) - np.log(a), np.nan)
    # Treatment intensity (inv intensity, pre-period)
    df["inv_intensity"] = df["cap_inv_total_pre"] / df["EMPTOT_pre"]
    df["log_inv_intensity"] = np.where(df["inv_intensity"] > 0, np.log(df["inv_intensity"]), np.nan)
    df = df.reset_index().rename(columns={"index": "ind_rev4"})
    df = df[df["n_eff_pre"].fillna(0) >= 10]

    def wls(df, y, x):
        sub = df.dropna(subset=[y, x])
        if len(sub) < 6:
            return None, sub
        X = sm.add_constant(sub[x])
        w = sub["n_eff_pre"].fillna(0).clip(lower=1)
        return sm.WLS(sub[y], X, weights=w).fit(), sub

    print("\n=== Productivity DiD: long-diff on log(inv intensity) ===")
    rows = []
    for ycol, label in [
        ("d_log_labor_prod",     "Labor productivity (VA / L)"),
        ("d_log_labor_prod_VBP", "Labor productivity (VBP / L)"),
        ("d_log_VBP",            "Gross output (VBP)"),
        ("d_log_VA",             "Value added (VA)"),
    ]:
        m, _ = wls(df, ycol, "log_inv_intensity")
        if m is None: continue
        rows.append({"outcome": label, "beta": m.params["log_inv_intensity"],
                     "se": m.bse["log_inv_intensity"],
                     "p": m.pvalues["log_inv_intensity"],
                     "R2": m.rsquared, "n": int(m.nobs),
                     "mean_dy": m.model.endog.mean()})
    tbl = pd.DataFrame(rows)
    print(tbl.round(3).to_string(index=False))
    tbl.to_csv(OUT / "productivity_diff_table.csv", index=False)

    # Pre-trend test for productivity
    print("\n=== Productivity DiD pre-trend (within pre) ===")
    pre_early = industry_mean([2010, 2011], outcomes)
    pre_late  = industry_mean([2013],       outcomes)
    pt = pre_early.add_suffix("_a").join(pre_late.add_suffix("_b"), how="outer")
    pt["labor_prod_a"]  = pt["VA_a"]  / pt["EMPTOT_a"]
    pt["labor_prod_b"]  = pt["VA_b"]  / pt["EMPTOT_b"]
    pt["d_log_labor_prod"] = np.where(
        (pt["labor_prod_a"]>0) & (pt["labor_prod_b"]>0),
        np.log(pt["labor_prod_b"]) - np.log(pt["labor_prod_a"]), np.nan)
    pt["d_log_VBP"] = np.where((pt["VBP_a"]>0) & (pt["VBP_b"]>0),
                                np.log(pt["VBP_b"]) - np.log(pt["VBP_a"]), np.nan)
    pt = pt.reset_index().rename(columns={"index":"ind_rev4"})
    pt = pt.merge(df[["ind_rev4","log_inv_intensity","n_eff_pre"]], on="ind_rev4", how="inner")
    for ycol, label in [("d_log_labor_prod","Labor productivity"),
                        ("d_log_VBP",       "Gross output")]:
        m, _ = wls(pt, ycol, "log_inv_intensity")
        if m is None: continue
        print(f"  PRE-TREND  {label}: β={m.params['log_inv_intensity']:+.3f}  "
              f"se={m.bse['log_inv_intensity']:.3f}  "
              f"p={m.pvalues['log_inv_intensity']:.3f}  n={int(m.nobs)}")

    # Placebo (fake 2011)
    print("\n=== Productivity DiD placebo (fake reform 2011) ===")
    pl_pre  = industry_mean([2008, 2009, 2010], outcomes)
    pl_post = industry_mean([2011, 2013],        outcomes)
    pl = pl_pre.add_suffix("_a").join(pl_post.add_suffix("_b"), how="outer")
    pl["labor_prod_a"]  = pl["VA_a"]  / pl["EMPTOT_a"]
    pl["labor_prod_b"]  = pl["VA_b"]  / pl["EMPTOT_b"]
    pl["d_log_labor_prod"] = np.where(
        (pl["labor_prod_a"]>0) & (pl["labor_prod_b"]>0),
        np.log(pl["labor_prod_b"]) - np.log(pl["labor_prod_a"]), np.nan)
    pl["d_log_VBP"] = np.where((pl["VBP_a"]>0) & (pl["VBP_b"]>0),
                                np.log(pl["VBP_b"]) - np.log(pl["VBP_a"]), np.nan)
    pl = pl.reset_index().rename(columns={"index":"ind_rev4"})
    pl = pl.merge(df[["ind_rev4","log_inv_intensity","n_eff_pre"]], on="ind_rev4", how="inner")
    for ycol, label in [("d_log_labor_prod","Labor productivity"),
                        ("d_log_VBP",       "Gross output")]:
        m, _ = wls(pl, ycol, "log_inv_intensity")
        if m is None: continue
        print(f"  PLACEBO   {label}: β={m.params['log_inv_intensity']:+.3f}  "
              f"se={m.bse['log_inv_intensity']:.3f}  "
              f"p={m.pvalues['log_inv_intensity']:.3f}  n={int(m.nobs)}")

    # Scatter for productivity
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (ycol, label) in zip(axes, [
        ("d_log_labor_prod",     "Δ log Labor productivity (VA/L)"),
        ("d_log_labor_prod_VBP", "Δ log Labor productivity (VBP/L)"),
    ]):
        sub = df.dropna(subset=[ycol, "log_inv_intensity"])
        if sub.empty: continue
        s = sub["n_eff_pre"].fillna(1).clip(lower=5)
        ax.scatter(sub["log_inv_intensity"], sub[ycol], alpha=0.5, s=s, color="darkorange")
        m, _ = wls(df, ycol, "log_inv_intensity")
        if m is not None:
            xx = np.linspace(sub["log_inv_intensity"].min(), sub["log_inv_intensity"].max(), 50)
            yy = m.params["const"] + m.params["log_inv_intensity"] * xx
            ax.plot(xx, yy, "k-", linewidth=1.5)
            ax.text(0.05, 0.95,
                    f"β = {m.params['log_inv_intensity']:+.3f} ({m.bse['log_inv_intensity']:.3f})\n"
                    f"p = {m.pvalues['log_inv_intensity']:.3f}, n = {int(m.nobs)}",
                    transform=ax.transAxes, va="top", fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"))
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("log(pre-period investment / EMPTOT)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.suptitle("Productivity outcomes: long-difference DiD\n"
                 f"Pre: {PRE_YEARS}    Post: {POST_YEARS}    Bubble size ∝ pre-period N",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "D3_productivity_scatter.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {FIGS / 'D3_productivity_scatter.png'}")


if __name__ == "__main__":
    main()
