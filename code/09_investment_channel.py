"""Investment channel test (per user hypothesis):

  Reform → investment inflow ↓ → investment-sensitive industries cut capex →
  hiring freeze → employment ↓

Treatment intensity: industry's pre-reform investment intensity (CB_inv / EMPTOT).
First stage:  Δ log Investment_industry on log(inv_intensity_industry).  Expect β < 0.
Second stage: Δ log Employment_industry  on log(inv_intensity_industry).  Expect β < 0.
Third stage:  Δ log VBP and Δ log Stock.                                  Expect β < 0.

Windows:
  Pre:  2010, 2011, 2013   (drop 2008/2009 schema bugs, 2012 low response)
  Post: 2016, 2017         (drop 2014/2015 transition)

Capital field aliasing across schema break:
  Investment flow (per row, CLP nominal):
    old (2008-2014): CBNEDI + CBNMAQ + CBNVEH + CBNSOF + CBNTER + CBNMUE + CBNOAT + CBNOTRI
    new (2015-2017): CBEDI  + CBMAQ  + CBVEH  + CBSOF  + CBTER          + CBOAT  + CBOTRI
  Capital stock (book/replacement value, per row):
    old: VBUEDI + VBUMAQ + VBUVEH + VBUSOF + VBUTER + VBUMUE + VBURET
    new: VREDI  + VRMAQ  + VRVEH  + VRSOF  + VRTER          + VROAT  + VROTRI
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

PRE_YEARS = [2010, 2011, 2013]
POST_YEARS = [2016, 2017]

OLD_INV = ["CBNEDI", "CBNMAQ", "CBNVEH", "CBNSOF", "CBNTER", "CBNMUE", "CBNOAT", "CBNOTRI"]
NEW_INV = ["CBEDI",  "CBMAQ",  "CBVEH",  "CBSOF",  "CBTER",            "CBOAT",  "CBOTRI"]
OLD_STK = ["VBUEDI", "VBUMAQ", "VBUVEH", "VBUSOF", "VBUTER", "VBUMUE", "VBURET"]
NEW_STK = ["VREDI",  "VRMAQ",  "VRVEH",  "VRSOF",  "VRTER",            "VROAT",  "VROTRI"]


def add_capital_aggregates(pool: pd.DataFrame) -> pd.DataFrame:
    """Add cap_inv_total and cap_stk_total columns to pool, using year-appropriate fields."""
    old_mask = pool["ANIO"] <= 2014
    new_mask = pool["ANIO"] >= 2015

    def sum_cols(df, cols):
        present = [c for c in cols if c in df.columns]
        return df[present].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)

    cap_inv = pd.Series(np.nan, index=pool.index, dtype=float)
    cap_inv.loc[old_mask] = sum_cols(pool.loc[old_mask], OLD_INV)
    cap_inv.loc[new_mask] = sum_cols(pool.loc[new_mask], NEW_INV)
    pool["cap_inv_total"] = cap_inv

    cap_stk = pd.Series(np.nan, index=pool.index, dtype=float)
    cap_stk.loc[old_mask] = sum_cols(pool.loc[old_mask], OLD_STK)
    cap_stk.loc[new_mask] = sum_cols(pool.loc[new_mask], NEW_STK)
    pool["cap_stk_total"] = cap_stk
    return pool


def industry_mean(pool, years, outcome_cols):
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
        tmp = pd.DataFrame({
            "ind": sub["ind_rev4"][ok].values,
            "vw": (vals[ok] * w[ok]).values,
            "w":  w[ok].values,
        })
        agg = tmp.groupby("ind").agg(num=("vw", "sum"), den=("w", "sum"))
        agg["wmean"] = agg["num"] / agg["den"]
        result[c] = agg["wmean"]
    result["n_eff"] = (sub.groupby("ind_rev4")["ind_weight"].sum()).reindex(industries)
    return result


def lr(df, y_col, x_col, weight_col="n_eff_pre"):
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


def main():
    pool = pd.read_parquet(DATA / "enia_pooled.parquet")
    pool = add_capital_aggregates(pool)
    print(f"Pool with capital aggregates: {pool.shape}")
    print(f"cap_inv_total non-null by year:")
    for y in sorted(pool["ANIO"].dropna().unique()):
        sub = pool[pool["ANIO"]==y]["cap_inv_total"]
        print(f"  {int(y)}: {sub.notna().sum()}/{len(sub)} non-null, "
              f"{(sub>0).sum()} positive, median nonzero={sub[sub>0].median():,.0f}")

    outcomes = ["cap_inv_total", "cap_stk_total", "EMPTOT", "VBP", "RENIMP", "REGPRDI", "REMHPRDI"]
    pre = industry_mean(pool, PRE_YEARS, outcomes)
    post = industry_mean(pool, POST_YEARS, outcomes)

    df = pre.add_suffix("_pre").join(post.add_suffix("_post"), how="outer")

    # log differences (skip non-positive)
    for c in outcomes:
        ppre = df[f"{c}_pre"]
        ppost = df[f"{c}_post"]
        valid = (ppre > 0) & (ppost > 0)
        df[f"d_log_{c}"] = np.where(valid, np.log(ppost) - np.log(ppre), np.nan)

    # Treatment intensities (computed from pre means)
    df["inv_intensity_KL"] = df["cap_inv_total_pre"] / df["EMPTOT_pre"]
    df["log_inv_intensity"] = np.where(df["inv_intensity_KL"] > 0, np.log(df["inv_intensity_KL"]), np.nan)
    df["inv_share_output"] = df["cap_inv_total_pre"] / df["VBP_pre"]
    df["log_inv_share"] = np.where(df["inv_share_output"] > 0, np.log(df["inv_share_output"]), np.nan)

    df = df.reset_index().rename(columns={"index": "ind_rev4"})
    df = df[df["n_eff_pre"].fillna(0) >= 10]
    print(f"\nIndustries with n_eff_pre>=10: {len(df)}")
    df.to_csv(OUT / "investment_channel_industry_data.csv", index=False)

    # Run regressions for each outcome on each treatment
    print("\n=== Long-difference DiD: treatment = log(investment / EMPTOT, pre-period) ===")
    rows = []
    test_set = [
        ("d_log_cap_inv_total",  "Investment flow (1st stage)"),
        ("d_log_cap_stk_total",  "Capital stock"),
        ("d_log_EMPTOT",         "Employment"),
        ("d_log_VBP",            "Gross output"),
        ("d_log_RENIMP",         "Corp income tax"),
        ("d_log_REGPRDI",        "Production wage"),
        ("d_log_REMHPRDI",       "Production hourly wage"),
    ]
    for ycol, name in test_set:
        model, sub = lr(df, ycol, "log_inv_intensity")
        if model is None:
            print(f"  {name}: skipped (n={len(sub)})"); continue
        b = model.params["log_inv_intensity"]
        se = model.bse["log_inv_intensity"]
        rows.append({"treatment": "log_inv_intensity (KL)", "outcome": name, "beta": b, "se": se,
                     "t_stat": b/se, "p": model.pvalues["log_inv_intensity"],
                     "R2": model.rsquared, "n": int(model.nobs),
                     "mean_dy": sub[ycol].mean()})

    print("\n=== Long-difference DiD: treatment = log(investment / VBP, pre-period) ===")
    for ycol, name in test_set:
        model, sub = lr(df, ycol, "log_inv_share")
        if model is None: continue
        b = model.params["log_inv_share"]
        se = model.bse["log_inv_share"]
        rows.append({"treatment": "log_inv_share (output)", "outcome": name, "beta": b, "se": se,
                     "t_stat": b/se, "p": model.pvalues["log_inv_share"],
                     "R2": model.rsquared, "n": int(model.nobs),
                     "mean_dy": sub[ycol].mean()})

    tbl = pd.DataFrame(rows)
    print(tbl.round(3).to_string(index=False))
    tbl.to_csv(OUT / "investment_channel_table.csv", index=False)

    # Plot: 6-panel scatter with treatment = log_inv_intensity
    plot_set = [
        ("d_log_cap_inv_total",  "Δ log Investment flow",     "darkblue",    "1st stage: do treated industries cut capex?"),
        ("d_log_cap_stk_total",  "Δ log Capital stock",       "darkorange",  "Capital stock follow-up"),
        ("d_log_EMPTOT",         "Δ log Employment",          "blue",        "Hiring freeze?"),
        ("d_log_VBP",            "Δ log VBP (gross output)",  "darkgreen",   "Output response"),
        ("d_log_REGPRDI",        "Δ log REGPRDI (prod wage)", "purple",      "Wage response"),
        ("d_log_RENIMP",         "Δ log RENIMP (corp tax)",   "darkred",     "Tax response"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (ycol, ylab, color, subtitle) in zip(axes.flat, plot_set):
        sub = df.dropna(subset=[ycol, "log_inv_intensity"])
        if sub.empty:
            ax.set_title(f"{ylab}\n(no data)"); continue
        size = sub["n_eff_pre"].fillna(1).clip(lower=5)
        ax.scatter(sub["log_inv_intensity"], sub[ycol], alpha=0.45, s=size, color=color)
        model, _ = lr(df, ycol, "log_inv_intensity")
        if model is not None:
            xx = np.linspace(sub["log_inv_intensity"].min(), sub["log_inv_intensity"].max(), 50)
            yy = model.params["const"] + model.params["log_inv_intensity"] * xx
            ax.plot(xx, yy, "k-", linewidth=1.5)
            b = model.params["log_inv_intensity"]
            se = model.bse["log_inv_intensity"]
            p = model.pvalues["log_inv_intensity"]
            ax.text(0.05, 0.95,
                    f"β = {b:+.3f} ({se:.3f})\np = {p:.3f}, n = {int(model.nobs)}",
                    transform=ax.transAxes, va="top", fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"))
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("log(pre-period investment / EMPTOT)")
        ax.set_ylabel(ylab)
        ax.set_title(subtitle, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Investment-channel DiD: industry-level Δ outcome vs baseline investment intensity\n"
        f"Pre: {PRE_YEARS}  Post: {POST_YEARS}    Bubble size ∝ pre-period N    "
        "Expected sign for treated industries: β < 0",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "B2_investment_channel.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {FIGS / 'B2_investment_channel.png'}")


if __name__ == "__main__":
    main()
