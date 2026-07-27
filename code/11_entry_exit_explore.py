"""Direction 1 feasibility exploration: industry establishment count as
outcome proxy for firm exit/entry.

Quick visual + tabular exploration before committing to full pipeline:

  1. Plot ENIA total establishment count by year (aggregate trend)
  2. Plot industry-year N_establishments evolution by tertile of baseline
     effective tax burden (RENIMP / VBP pre-reform).
  3. Same for tertile of baseline capital intensity (CB_inv / EMPTOT).
  4. Long-difference: Δ log N_establishments ~ baseline characteristics
  5. Run the same pre-trend + placebo test framework.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "output"
FIGS = OUT / "figs"


def main():
    pool = pd.read_parquet(DATA / "enia_pooled.parquet")
    # establishment counts: weighted sum of ind_weight per (industry, year)
    sub = pool[pool["ind_rev4"].notna() & pool["ANIO"].notna()]
    iy = (sub.groupby(["ind_rev4", "ANIO"])["ind_weight"].sum()
             .rename("n_est").reset_index())

    # 1. Aggregate ENIA total establishment count by year
    by_year = iy.groupby("ANIO")["n_est"].sum().rename("total_n").reset_index()
    print("\n=== Aggregate ENIA establishment count by year ===")
    print(by_year.to_string(index=False))

    # plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(by_year["ANIO"], by_year["total_n"], color="#4477AA")
    ax.axvline(2014, color="red", linestyle="--", alpha=0.6, label="2014 reform")
    ax.set_xlabel("Year"); ax.set_ylabel("Total establishments (weighted)")
    ax.set_title("ENIA total establishment count by year")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIGS / "C1_total_establishments_by_year.png", dpi=140)
    plt.close(fig)

    # 2. Build baseline characteristics per industry from pre-period (2010, 2011, 2013)
    pre_years = [2010, 2011, 2013]
    pre = pool[pool["ANIO"].isin(pre_years) & pool["ind_rev4"].notna()].copy()
    pre["RENIMP"] = pd.to_numeric(pre["RENIMP"], errors="coerce")
    pre["VBP"]    = pd.to_numeric(pre["VBP"],    errors="coerce")
    pre["EMPTOT"] = pd.to_numeric(pre["EMPTOT"], errors="coerce")
    # Add capital inv aggregate
    OLD_INV = ["CBNEDI", "CBNMAQ", "CBNVEH", "CBNSOF", "CBNTER", "CBNMUE", "CBNOAT", "CBNOTRI"]
    present = [c for c in OLD_INV if c in pre.columns]
    cap_inv = pre[present].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    pre["cap_inv_total"] = cap_inv

    def wmean(g, col):
        v = pd.to_numeric(g[col], errors="coerce")
        w = pd.to_numeric(g["ind_weight"], errors="coerce")
        ok = v.notna() & w.notna() & (w > 0)
        if ok.sum() == 0: return np.nan
        return float((v[ok] * w[ok]).sum() / w[ok].sum())

    baseline = []
    for ind, g in pre.groupby("ind_rev4"):
        baseline.append({
            "ind_rev4":  ind,
            "n_pre":     g["ind_weight"].sum(),
            "RENIMP":    wmean(g, "RENIMP"),
            "VBP":       wmean(g, "VBP"),
            "EMPTOT":    wmean(g, "EMPTOT"),
            "cap_inv":   wmean(g, "cap_inv_total"),
        })
    bdf = pd.DataFrame(baseline)
    bdf["tax_burden"]    = bdf["RENIMP"] / bdf["VBP"]
    bdf["inv_intensity"] = bdf["cap_inv"] / bdf["EMPTOT"]
    # log treatments
    for c in ["tax_burden", "inv_intensity"]:
        bdf[f"log_{c}"] = np.where(bdf[c] > 0, np.log(bdf[c]), np.nan)
    # filter
    bdf = bdf[bdf["n_pre"] >= 10]
    print(f"\nIndustries with baseline n_pre>=10: {len(bdf)}")

    # 3. industry-year N_establishments → log; merge baseline → tertile plots
    iy_log = iy.merge(bdf[["ind_rev4", "log_tax_burden", "log_inv_intensity", "n_pre"]],
                      on="ind_rev4", how="inner")
    iy_log = iy_log[iy_log["n_pre"] >= 10]

    # Tertile by tax burden
    for treat_col, label in [("log_tax_burden", "effective tax burden"),
                             ("log_inv_intensity", "investment intensity")]:
        tertiles = pd.qcut(bdf.set_index("ind_rev4")[treat_col].dropna(), q=3,
                           labels=["low", "mid", "high"])
        iy_log_t = iy_log.copy()
        iy_log_t["tertile"] = iy_log_t["ind_rev4"].map(tertiles)
        iy_log_t = iy_log_t.dropna(subset=["tertile"])
        # Build per-tertile yearly mean log(N_est)
        # index each industry at 2013 (last pre-reform clean year)
        ind_pivot = iy_log_t.pivot_table(index="ind_rev4", columns="ANIO", values="n_est", aggfunc="sum")
        ind_pivot = ind_pivot.div(ind_pivot[2013], axis=0) * 100  # index to 2013=100
        # add tertile column
        ind_pivot["tertile"] = ind_pivot.index.map(tertiles)
        tertile_year = (ind_pivot.dropna(subset=["tertile"])
                        .groupby("tertile")[[y for y in ind_pivot.columns if y != "tertile"]]
                        .median().T)

        fig, ax = plt.subplots(figsize=(9, 5))
        for tert, color in zip(["low", "mid", "high"], ["#4477AA", "#888888", "#CC3311"]):
            if tert in tertile_year.columns:
                ax.plot(tertile_year.index, tertile_year[tert], marker="o",
                        label=f"{tert} {label}", color=color)
        ax.axvline(2014, color="red", linestyle="--", alpha=0.6, label="2014 reform")
        ax.axhline(100, color="gray", linewidth=0.5)
        ax.set_xlabel("Year"); ax.set_ylabel("N_establishments index (2013=100)")
        ax.set_title(f"Establishment count by tertile of baseline {label}")
        ax.legend(); ax.grid(alpha=0.3)
        fname = f"C2_N_est_by_tertile_{label.replace(' ', '_')}.png"
        fig.tight_layout(); fig.savefig(FIGS / fname, dpi=140); plt.close(fig)
        print(f"  wrote {fname}")

    # 4. Long-difference: Δ log(N_est) on baseline tax_burden
    pre_window  = [2010, 2011, 2013]
    post_window = [2016, 2017]
    pre_n  = iy[iy["ANIO"].isin(pre_window)].groupby("ind_rev4")["n_est"].mean().rename("n_pre_mean")
    post_n = iy[iy["ANIO"].isin(post_window)].groupby("ind_rev4")["n_est"].mean().rename("n_post_mean")
    n_diff = pre_n.to_frame().join(post_n.to_frame(), how="outer")
    n_diff["d_log_n_est"] = np.where(
        (n_diff["n_pre_mean"] > 0) & (n_diff["n_post_mean"] > 0),
        np.log(n_diff["n_post_mean"]) - np.log(n_diff["n_pre_mean"]),
        np.nan,
    )
    merged = n_diff.join(bdf.set_index("ind_rev4"), how="inner")
    merged = merged[merged["n_pre"] >= 10]

    print(f"\n=== Direction 1: Long-difference Δ log(N_establishments) ===")
    print(f"Industries: {len(merged)}")
    print(f"Mean Δ log N_est across industries: {merged['d_log_n_est'].mean():.3f}")
    print(f"(positive = industry grew, negative = industry shrank)")

    # Main spec
    rows = []
    for treat, label in [("log_tax_burden", "Tax burden"), ("log_inv_intensity", "Investment intensity")]:
        sub = merged.dropna(subset=["d_log_n_est", treat]).copy()
        X = sm.add_constant(sub[treat])
        w = sub["n_pre"].fillna(0).clip(lower=1)
        m = sm.WLS(sub["d_log_n_est"], X, weights=w).fit()
        rows.append({"spec": "MAIN", "treatment": label,
                     "beta": m.params[treat], "se": m.bse[treat],
                     "p": m.pvalues[treat], "R2": m.rsquared, "n": int(m.nobs),
                     "mean_dy": sub["d_log_n_est"].mean()})

    # Pre-trend (within pre: 2013 vs avg(2010,2011))
    pre_early = iy[iy["ANIO"].isin([2010, 2011])].groupby("ind_rev4")["n_est"].mean().rename("n_pe")
    pre_late  = iy[iy["ANIO"]==2013].groupby("ind_rev4")["n_est"].mean().rename("n_pl")
    pt = pre_early.to_frame().join(pre_late.to_frame(), how="outer")
    pt["d_log_n_est"] = np.where((pt["n_pe"] > 0) & (pt["n_pl"] > 0),
                                  np.log(pt["n_pl"]) - np.log(pt["n_pe"]), np.nan)
    pt_merged = pt.join(bdf.set_index("ind_rev4"), how="inner")
    pt_merged = pt_merged[pt_merged["n_pre"] >= 10]
    for treat, label in [("log_tax_burden", "Tax burden"), ("log_inv_intensity", "Investment intensity")]:
        sub = pt_merged.dropna(subset=["d_log_n_est", treat]).copy()
        X = sm.add_constant(sub[treat])
        w = sub["n_pre"].fillna(0).clip(lower=1)
        m = sm.WLS(sub["d_log_n_est"], X, weights=w).fit()
        rows.append({"spec": "PRE-TREND", "treatment": label,
                     "beta": m.params[treat], "se": m.bse[treat],
                     "p": m.pvalues[treat], "R2": m.rsquared, "n": int(m.nobs),
                     "mean_dy": sub["d_log_n_est"].mean()})

    # Placebo 2011 (fake reform)
    pl_pre  = iy[iy["ANIO"].isin([2008, 2009, 2010])].groupby("ind_rev4")["n_est"].mean().rename("npre")
    pl_post = iy[iy["ANIO"].isin([2011, 2013])].groupby("ind_rev4")["n_est"].mean().rename("npost")
    pl = pl_pre.to_frame().join(pl_post.to_frame(), how="outer")
    pl["d_log_n_est"] = np.where((pl["npre"] > 0) & (pl["npost"] > 0),
                                  np.log(pl["npost"]) - np.log(pl["npre"]), np.nan)
    pl_merged = pl.join(bdf.set_index("ind_rev4"), how="inner")
    pl_merged = pl_merged[pl_merged["n_pre"] >= 10]
    for treat, label in [("log_tax_burden", "Tax burden"), ("log_inv_intensity", "Investment intensity")]:
        sub = pl_merged.dropna(subset=["d_log_n_est", treat]).copy()
        X = sm.add_constant(sub[treat])
        w = sub["n_pre"].fillna(0).clip(lower=1)
        m = sm.WLS(sub["d_log_n_est"], X, weights=w).fit()
        rows.append({"spec": "PLACEBO (fake 2011)", "treatment": label,
                     "beta": m.params[treat], "se": m.bse[treat],
                     "p": m.pvalues[treat], "R2": m.rsquared, "n": int(m.nobs),
                     "mean_dy": sub["d_log_n_est"].mean()})

    tbl = pd.DataFrame(rows)
    print(tbl.round(3).to_string(index=False))
    tbl.to_csv(OUT / "entry_exit_long_diff.csv", index=False)


if __name__ == "__main__":
    main()
