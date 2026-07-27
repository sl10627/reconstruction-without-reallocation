"""Step A: descriptive time-series plots.

Uses a coarse capital-intensity proxy (industry-level VA per worker in 2013)
to split industries into tertiles. Then plots outcome trends 2008-2017 by tertile.

Figures saved to output/figs/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "output" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

iy = pd.read_parquet(DATA / "enia_industry_year.parquet")
print(f"industry_year loaded: {iy.shape}")
print(f"industries: {iy['ind_rev4'].nunique()}, years: {sorted(iy['ANIO'].unique())}")

# --- Coarse capital-intensity proxy: pre-reform avg VBP per worker ---
# VA only exists 2014+, so use VBP (gross output) per EMPTOT.
# Pre-reform years with reliable VBP: 2008, 2009, 2010, 2011, 2013 (avg).
pre = iy[iy["ANIO"].isin([2008, 2009, 2010, 2011, 2013])].copy()
pre["k_proxy_per_year"] = pre["VBP_wsum"] / pre["EMPTOT_wsum"]
base = pre.groupby("ind_rev4").agg(
    k_proxy=("k_proxy_per_year", "mean"),
    n_establishments=("n_establishments", "mean"),
).reset_index()
base = base[(base["n_establishments"] >= 5) & base["k_proxy"].notna() & (base["k_proxy"] > 0)]
print(f"\nIndustries with valid pre-reform VBP/EMPTOT proxy (avg n>=5): {len(base)}")
# tertile assignment
base["tertile"] = pd.qcut(base["k_proxy"], q=3, labels=["low", "mid", "high"])
print(base["tertile"].value_counts().sort_index())
print(f"\nproxy distribution by tertile:")
print(base.groupby("tertile", observed=True)["k_proxy"].agg(["min", "median", "max"]))

industry_tertile = base.set_index("ind_rev4")["tertile"].to_dict()
iy["tertile"] = iy["ind_rev4"].map(industry_tertile)

# ---- aggregate to (tertile, year) cells for plotting ----
# weighted mean across industries within each tertile, weighting by establishment count
def tertile_year_mean(df, outcome_col, weight_col="n_establishments"):
    out = df.dropna(subset=[outcome_col, "tertile"]).copy()
    grouped = out.groupby(["tertile", "ANIO"], observed=True).apply(
        lambda g: (g[outcome_col] * g[weight_col]).sum() / g[weight_col].sum()
        if g[weight_col].sum() > 0 else np.nan,
        include_groups=False,
    )
    return grouped.unstack("tertile")  # rows=year, cols=tertile

REFORM_YEAR = 2014

def plot_outcome(outcome_col, title, ylab, fname, year_range=None):
    df = tertile_year_mean(iy, outcome_col)
    if year_range:
        df = df.loc[df.index.isin(year_range)]
    fig, ax = plt.subplots(figsize=(9, 5))
    for tert, color in zip(["low", "mid", "high"], ["#4477AA", "#888888", "#CC3311"]):
        if tert in df.columns:
            ax.plot(df.index, df[tert], marker="o", label=f"{tert} cap-intensity", color=color)
    ax.axvline(REFORM_YEAR, color="black", linestyle="--", alpha=0.5, label="2014 reform")
    ax.set_xlabel("Year")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=140)
    plt.close(fig)
    print(f"  wrote {fname}")
    return df

# ---- A1. Sample size: n_establishments by year, summed across industries ----
n_year = iy.groupby("ANIO")["n_establishments"].sum()
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(n_year.index, n_year.values, color="#4477AA")
ax.axvline(REFORM_YEAR, color="black", linestyle="--", alpha=0.5, label="2014 reform")
ax.set_xlabel("Year"); ax.set_ylabel("# establishments"); ax.set_title("ENIA sample size by year")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(FIGS / "A1_sample_size.png", dpi=140); plt.close(fig)
print("  wrote A1_sample_size.png")

# ---- A2. Mean total employment per establishment, by tertile ----
print("\nA2. EMPTOT (mean per establishment)")
emp_df = plot_outcome("EMPTOT_wmean", "Mean total employment per establishment",
                       "employees", "A2_EMPTOT_wmean_by_tertile.png")
print(emp_df.round(1))

# ---- A3. RENIMP (tax paid, mean per establishment) ----
print("\nA3. RENIMP (mean per establishment)")
ren_df = plot_outcome("RENIMP_wmean", "Mean corporate income tax (RENIMP) per establishment",
                       "CLP (nominal)", "A3_RENIMP_wmean_by_tertile.png")
print(ren_df.round(0))

# ---- A4. CONIMP (other tax/contribution, sanity placebo) ----
print("\nA4. CONIMP (mean per establishment)")
con_df = plot_outcome("CONIMP_wmean", "Mean property/social tax (CONIMP) per establishment",
                       "CLP (nominal)", "A4_CONIMP_wmean_by_tertile.png")
print(con_df.round(0))

# ---- A5. VBP (gross output, 2013+ only) ----
print("\nA5. VBP (mean per establishment, 2013+)")
vbp_df = plot_outcome("VBP_wmean", "Mean gross output (VBP) per establishment",
                       "CLP (nominal)", "A5_VBP_wmean_by_tertile.png", year_range=range(2013, 2018))
print(vbp_df.round(0))

# ---- A6. VA (value added, 2013+ only) ----
print("\nA6. VA (mean per establishment, 2013+)")
va_df = plot_outcome("VA_wmean", "Mean value added (VA) per establishment",
                      "CLP (nominal)", "A6_VA_wmean_by_tertile.png", year_range=range(2013, 2018))
print(va_df.round(0))

# ---- A7. Normalized: indexed to 2013 = 100 for each tertile ----
def plot_indexed(outcome_col, title, fname, year_range=None):
    df = tertile_year_mean(iy, outcome_col)
    if year_range:
        df = df.loc[df.index.isin(year_range)]
    if 2013 not in df.index:
        return
    base = df.loc[2013]
    idx = df.div(base) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    for tert, color in zip(["low", "mid", "high"], ["#4477AA", "#888888", "#CC3311"]):
        if tert in idx.columns:
            ax.plot(idx.index, idx[tert], marker="o", label=f"{tert} cap-intensity", color=color)
    ax.axvline(REFORM_YEAR, color="black", linestyle="--", alpha=0.5, label="2014 reform")
    ax.axhline(100, color="gray", linewidth=0.5)
    ax.set_xlabel("Year"); ax.set_ylabel(f"index, 2013=100"); ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / fname, dpi=140); plt.close(fig)
    print(f"  wrote {fname}")

print("\nA7-A9. Indexed (2013=100):")
plot_indexed("EMPTOT_wmean", "Employment per establishment (2013=100)", "A7_EMPTOT_indexed.png")
plot_indexed("RENIMP_wmean", "RENIMP per establishment (2013=100)", "A8_RENIMP_indexed.png")
plot_indexed("VA_wmean", "VA per establishment (2013=100)", "A9_VA_indexed.png", year_range=range(2013, 2018))

print(f"\nAll figures saved to {FIGS}")
