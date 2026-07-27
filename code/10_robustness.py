"""Three robustness checks for the investment channel result.

A. Pre-trend test: do treated industries already differ in trajectory before
   reform? Construct a "pre-only" long-diff: late-pre (2013) vs early-pre
   (2010, 2011). If treatment intensity already correlates with this within-pre
   change, the main result is contaminated by pre-trend.

B. Placebo treatment date: pretend reform was in 2011 instead of 2014. Use
   2008-2010 as fake pre, 2011-2013 as fake post. Build treatment intensity
   from fake pre (2008-2010). If significant, the main β is spurious.

C. Industry-level macro controls: add export share, pre-period output trend,
   and sector (2-digit) fixed effects to the main regression. If β survives,
   it's not just macro confounding.

Outputs:
  output/robustness_pretrend.csv
  output/robustness_placebo.csv
  output/robustness_macrocontrols.csv
  output/figs/B3_robustness.png
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
NEW_INV = ["CBEDI",  "CBMAQ",  "CBVEH",  "CBSOF",  "CBTER",            "CBOAT",  "CBOTRI"]
OLD_STK = ["VBUEDI", "VBUMAQ", "VBUVEH", "VBUSOF", "VBUTER", "VBUMUE", "VBURET"]
NEW_STK = ["VREDI",  "VRMAQ",  "VRVEH",  "VRSOF",  "VRTER",            "VROAT",  "VROTRI"]


def add_capital_aggregates(pool):
    old = pool["ANIO"] <= 2014
    new = pool["ANIO"] >= 2015
    def s(df, cols):
        present = [c for c in cols if c in df.columns]
        return df[present].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    pool["cap_inv_total"] = np.nan
    pool.loc[old, "cap_inv_total"] = s(pool.loc[old], OLD_INV)
    pool.loc[new, "cap_inv_total"] = s(pool.loc[new], NEW_INV)
    pool["cap_stk_total"] = np.nan
    pool.loc[old, "cap_stk_total"] = s(pool.loc[old], OLD_STK)
    pool.loc[new, "cap_stk_total"] = s(pool.loc[new], NEW_STK)
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
        result[c] = agg["num"] / agg["den"]
    result["n_eff"] = (sub.groupby("ind_rev4")["ind_weight"].sum()).reindex(industries)
    return result


def long_diff(pre, post, outcomes):
    """Build per-industry log-diff frame."""
    df = pre.add_suffix("_pre").join(post.add_suffix("_post"), how="outer")
    for c in outcomes:
        a, b = df[f"{c}_pre"], df[f"{c}_post"]
        valid = (a > 0) & (b > 0)
        df[f"d_log_{c}"] = np.where(valid, np.log(b) - np.log(a), np.nan)
    df["inv_intensity"] = df["cap_inv_total_pre"] / df["EMPTOT_pre"]
    df["log_inv_intensity"] = np.where(df["inv_intensity"] > 0, np.log(df["inv_intensity"]), np.nan)
    return df.reset_index().rename(columns={"index": "ind_rev4"})


def wls(df, y_col, x_cols, weight_col="n_eff_pre"):
    cols = [c for c in (x_cols if isinstance(x_cols, list) else [x_cols])]
    sub = df.dropna(subset=[y_col] + cols).copy()
    if len(sub) < 6:
        return None, sub
    X = sm.add_constant(sub[cols])
    w = sub[weight_col].fillna(0).clip(lower=1) if weight_col in sub.columns else None
    return sm.WLS(sub[y_col], X, weights=w).fit() if w is not None else sm.OLS(sub[y_col], X).fit(), sub


def summarize(model, x_name, label):
    if model is None: return None
    return {
        "spec": label,
        "x": x_name,
        "beta": model.params[x_name],
        "se": model.bse[x_name],
        "t": model.params[x_name] / model.bse[x_name],
        "p": model.pvalues[x_name],
        "R2": model.rsquared,
        "n": int(model.nobs),
    }


def main():
    pool = pd.read_parquet(DATA / "enia_pooled.parquet")
    pool = add_capital_aggregates(pool)
    outcomes = ["cap_inv_total", "cap_stk_total", "EMPTOT", "VBP", "RENIMP", "REGPRDI", "REMHPRDI", "EXPVAL"]

    # ---------- Main spec re-run (for reference) ----------
    pre_main  = industry_mean(pool, [2010, 2011, 2013], outcomes)
    post_main = industry_mean(pool, [2016, 2017],       outcomes)
    df_main = long_diff(pre_main, post_main, outcomes)
    df_main = df_main[df_main["n_eff_pre"].fillna(0) >= 10]
    print(f"Main spec: {len(df_main)} industries")

    main_rows = []
    test_outcomes = [
        ("d_log_cap_inv_total", "Investment flow"),
        ("d_log_cap_stk_total", "Capital stock"),
        ("d_log_EMPTOT",        "Employment"),
        ("d_log_VBP",           "Output"),
        ("d_log_REGPRDI",       "Production wage"),
    ]
    for ycol, name in test_outcomes:
        m, _ = wls(df_main, ycol, "log_inv_intensity")
        r = summarize(m, "log_inv_intensity", f"MAIN {name}")
        if r: main_rows.append({**r, "outcome": name})
    print("\n=== Main spec (reference) ===")
    print(pd.DataFrame(main_rows).round(3).to_string(index=False))

    # ---------- A. Pre-trend test ----------
    # Treat 2010, 2011 as "early pre", 2013 as "late pre".
    # Construct a within-pre long-diff with treatment built from the SAME
    # full pre window (2010, 2011, 2013) so treatment definition matches main spec.
    pre_early = industry_mean(pool, [2010, 2011], outcomes)
    pre_late  = industry_mean(pool, [2013],       outcomes)
    df_pt = long_diff(pre_early, pre_late, outcomes)
    # Replace the inv_intensity that long_diff computed from this window with the
    # one from the FULL pre window (so we test the same treatment).
    df_pt = df_pt.merge(df_main[["ind_rev4", "log_inv_intensity"]].rename(columns={"log_inv_intensity": "log_inv_intensity_main"}),
                        on="ind_rev4", how="left")
    df_pt["log_inv_intensity"] = df_pt["log_inv_intensity_main"]
    df_pt = df_pt[df_pt["n_eff_pre"].fillna(0) >= 10]

    pt_rows = []
    for ycol, name in test_outcomes:
        m, _ = wls(df_pt, ycol, "log_inv_intensity")
        r = summarize(m, "log_inv_intensity", f"PRE-TREND {name}")
        if r: pt_rows.append({**r, "outcome": name})
    print("\n=== A. Pre-trend test (within pre: 2013 vs avg(2010,2011)) ===")
    print("If β here is significant, the main β is contaminated by pre-existing trend.")
    print(pd.DataFrame(pt_rows).round(3).to_string(index=False))
    pd.DataFrame(pt_rows).to_csv(OUT / "robustness_pretrend.csv", index=False)

    # ---------- B. Placebo treatment date ----------
    # Fake reform = 2011. Fake pre = 2008-2010, fake post = 2011, 2013.
    # Build treatment from fake pre.
    pl_pre  = industry_mean(pool, [2008, 2009, 2010], outcomes)
    pl_post = industry_mean(pool, [2011, 2013],        outcomes)
    df_pl = long_diff(pl_pre, pl_post, outcomes)
    df_pl = df_pl[df_pl["n_eff_pre"].fillna(0) >= 10]

    pl_rows = []
    for ycol, name in test_outcomes:
        m, _ = wls(df_pl, ycol, "log_inv_intensity")
        r = summarize(m, "log_inv_intensity", f"PLACEBO {name}")
        if r: pl_rows.append({**r, "outcome": name})
    print("\n=== B. Placebo: fake reform 2011 (pre=2008-2010, post=2011,2013) ===")
    print("If β here is significant, main β is spurious (correlation predates reform).")
    print(pd.DataFrame(pl_rows).round(3).to_string(index=False))
    pd.DataFrame(pl_rows).to_csv(OUT / "robustness_placebo.csv", index=False)

    # ---------- C. Macro controls ----------
    # Controls to add to main regression:
    #   - export_share = EXPVAL_pre / VBP_pre (export orientation)
    #   - pre_trend_VBP = pre-period log VBP slope (industry-specific trend)
    #   - sector_FE = 2-digit ISIC dummies
    # Build controls
    df_c = df_main.copy()
    # export share
    df_c["export_share_pre"] = df_c["EXPVAL_pre"] / df_c["VBP_pre"]
    df_c["log_export_share_pre"] = np.where(df_c["export_share_pre"] > 0, np.log(df_c["export_share_pre"]), np.nan)
    # pre-trend slope of log VBP across 2010, 2011, 2013 per industry
    pre_panel = pool[(pool["ANIO"].isin([2010, 2011, 2013])) & (pool["ind_rev4"].notna())]
    industry_year_vbp = (pre_panel.assign(vbp_w=lambda d: pd.to_numeric(d["VBP"], errors="coerce") * d["ind_weight"],
                                          w=lambda d: d["ind_weight"])
                        .groupby(["ind_rev4", "ANIO"]).agg(vbp=("vbp_w", "sum"), w=("w", "sum"))
                        .assign(vbp_mean=lambda d: d["vbp"] / d["w"])
                        .reset_index())
    industry_year_vbp["log_vbp"] = np.log(industry_year_vbp["vbp_mean"].where(industry_year_vbp["vbp_mean"] > 0))

    slopes = {}
    for ind, g in industry_year_vbp.dropna().groupby("ind_rev4"):
        if len(g) >= 2:
            x = g["ANIO"].values - g["ANIO"].mean()
            y = g["log_vbp"].values
            if np.std(x) > 0:
                slopes[ind] = np.polyfit(x, y, 1)[0]
    df_c["pre_trend_VBP"] = df_c["ind_rev4"].map(slopes)
    # 2-digit ISIC sector
    df_c["sector2"] = df_c["ind_rev4"].astype(str).str[:2]

    mc_rows = []
    for ycol, name in test_outcomes:
        # baseline
        m0, _ = wls(df_c, ycol, "log_inv_intensity")
        if m0: mc_rows.append({"outcome": name, "spec": "0. main", **{k: m0.__dict__.get(k) for k in []},
                              "beta": m0.params["log_inv_intensity"], "se": m0.bse["log_inv_intensity"],
                              "p": m0.pvalues["log_inv_intensity"], "R2": m0.rsquared, "n": int(m0.nobs)})
        # add export share
        m1, _ = wls(df_c, ycol, ["log_inv_intensity", "log_export_share_pre"])
        if m1: mc_rows.append({"outcome": name, "spec": "1. +export_share",
                              "beta": m1.params["log_inv_intensity"], "se": m1.bse["log_inv_intensity"],
                              "p": m1.pvalues["log_inv_intensity"], "R2": m1.rsquared, "n": int(m1.nobs)})
        # add pre-trend
        m2, _ = wls(df_c, ycol, ["log_inv_intensity", "log_export_share_pre", "pre_trend_VBP"])
        if m2: mc_rows.append({"outcome": name, "spec": "2. +pre_trend",
                              "beta": m2.params["log_inv_intensity"], "se": m2.bse["log_inv_intensity"],
                              "p": m2.pvalues["log_inv_intensity"], "R2": m2.rsquared, "n": int(m2.nobs)})
        # add sector-2-digit FE
        sub = df_c.dropna(subset=[ycol, "log_inv_intensity", "log_export_share_pre", "pre_trend_VBP"]).copy()
        if len(sub) >= 10:
            sector_dummies = pd.get_dummies(sub["sector2"], prefix="sec", drop_first=True, dtype=float)
            X = pd.concat([sub[["log_inv_intensity", "log_export_share_pre", "pre_trend_VBP"]], sector_dummies], axis=1)
            X = sm.add_constant(X)
            w = sub["n_eff_pre"].fillna(0).clip(lower=1)
            try:
                m3 = sm.WLS(sub[ycol], X, weights=w).fit()
                mc_rows.append({"outcome": name, "spec": "3. +sector2FE",
                              "beta": m3.params["log_inv_intensity"], "se": m3.bse["log_inv_intensity"],
                              "p": m3.pvalues["log_inv_intensity"], "R2": m3.rsquared, "n": int(m3.nobs)})
            except Exception as e:
                print(f"  sector FE failed for {name}: {e}")
    print("\n=== C. Macro controls (stepwise) ===")
    mc_df = pd.DataFrame(mc_rows)[["outcome", "spec", "beta", "se", "p", "R2", "n"]]
    print(mc_df.round(3).to_string(index=False))
    mc_df.to_csv(OUT / "robustness_macrocontrols.csv", index=False)

    # ---------- Plot summary ----------
    # 4 outcomes × 4 specs (main, pre-trend, placebo, with macro+FE controls)
    spec_order = ["MAIN", "PRE-TREND", "PLACEBO", "+macro+sectorFE"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)
    for ax, (ycol, name) in zip(axes, test_outcomes):
        bars = {}
        # Pull values
        main = [r for r in main_rows if r["outcome"] == name]
        pt = [r for r in pt_rows if r["outcome"] == name]
        pl = [r for r in pl_rows if r["outcome"] == name]
        mc_last = mc_df[(mc_df["outcome"]==name) & (mc_df["spec"]=="3. +sector2FE")]
        coefs = {
            "MAIN": (main[0]["beta"], main[0]["se"], main[0]["p"]) if main else (np.nan, np.nan, np.nan),
            "PRE-TREND": (pt[0]["beta"], pt[0]["se"], pt[0]["p"]) if pt else (np.nan, np.nan, np.nan),
            "PLACEBO": (pl[0]["beta"], pl[0]["se"], pl[0]["p"]) if pl else (np.nan, np.nan, np.nan),
            "+macro+sectorFE": (mc_last["beta"].iloc[0], mc_last["se"].iloc[0], mc_last["p"].iloc[0]) if not mc_last.empty else (np.nan, np.nan, np.nan),
        }
        xs = list(coefs.keys())
        ys = [v[0] for v in coefs.values()]
        es = [1.96 * v[1] for v in coefs.values()]
        colors = ["black", "tab:orange", "tab:red", "tab:green"]
        ax.errorbar(range(len(xs)), ys, yerr=es, fmt="o", color="black", capsize=4)
        for j, (x, y, color) in enumerate(zip(xs, ys, colors)):
            ax.scatter(j, y, color=color, zorder=5, s=80)
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=30, ha="right", fontsize=9)
        ax.set_title(name, fontsize=11)
        ax.set_ylabel("β on log inv intensity")
        ax.grid(alpha=0.3)
    fig.suptitle("Robustness: coefficient on log(investment intensity) across specs (with 95% CI)\n"
                 "MAIN = real reform; PRE-TREND = within-pre placebo; PLACEBO = fake reform 2011; "
                 "+macro+sectorFE = add export share, pre-trend, 2-digit sector dummies",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "B3_robustness.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {FIGS / 'B3_robustness.png'}")


if __name__ == "__main__":
    main()
