"""Profile pre-earthquake (2008-2009) industry composition by region.

For each Chile region, compute:
  - Top 2-digit ISIC industries by establishment count
  - Top by total employment
  - Top by total output (VBP)

Output:
  output/industry_by_region_pre.csv
  output/figs/F1_industry_composition_by_region.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "output"
FIGS = OUT / "figs"

# ENIA REGION code → name (with MMI from our previous analysis)
ENIA_REGION_NAME = {
    1: "Tarapacá", 2: "Antofagasta", 3: "Atacama", 4: "Coquimbo",
    5: "Valparaíso", 6: "O'Higgins", 7: "Maule", 8: "Bío-Bío",
    9: "Araucanía", 10: "Los Lagos", 11: "Aysén", 12: "Magallanes",
    13: "Santiago Metro", 14: "Los Ríos", 15: "Arica y Parinacota",
}

# ISIC Rev.4 2-digit sector labels (manufacturing only)
ISIC_2D = {
    "10": "Food",
    "11": "Beverages",
    "12": "Tobacco",
    "13": "Textiles",
    "14": "Apparel",
    "15": "Leather",
    "16": "Wood",
    "17": "Paper/Pulp",
    "18": "Printing",
    "19": "Coke/Petroleum",
    "20": "Chemicals",
    "21": "Pharma",
    "22": "Rubber/Plastic",
    "23": "Non-metallic minerals",
    "24": "Basic metals (steel)",
    "25": "Fabricated metal",
    "26": "Electronics",
    "27": "Electrical eqpt",
    "28": "Machinery",
    "29": "Motor vehicles",
    "30": "Other transport",
    "31": "Furniture",
    "32": "Other manuf",
    "33": "Repair",
}


def main():
    pool = pd.read_parquet(DATA / "enia_pooled.parquet")
    # restrict to pre-earthquake: 2008, 2009
    pre = pool[pool["ANIO"].isin([2008, 2009]) & pool["ind_rev4"].notna()
               & pool["REGION"].notna()].copy()
    # 2-digit sector
    pre["sec2"] = pre["ind_rev4"].astype(str).str[:2]
    pre["sec_name"] = pre["sec2"].map(ISIC_2D).fillna("Other (sec=" + pre["sec2"] + ")")
    pre["REGION_int"] = pd.to_numeric(pre["REGION"], errors="coerce").astype("Int64")
    pre["region_name"] = pre["REGION_int"].map(ENIA_REGION_NAME)
    pre["EMPTOT"] = pd.to_numeric(pre["EMPTOT"], errors="coerce")
    pre["VBP"]    = pd.to_numeric(pre["VBP"],    errors="coerce")
    pre["w"] = pre["ind_weight"].fillna(0).clip(lower=0)

    # group by (region, sector)
    grp = pre.groupby(["REGION_int", "region_name", "sec_name"]).agg(
        n_est=("w", "sum"),
        emp =("EMPTOT", lambda s: (s.fillna(0) * pre.loc[s.index, "w"]).sum()),
        vbp =("VBP",    lambda s: (s.fillna(0) * pre.loc[s.index, "w"]).sum()),
    ).reset_index()

    # within each region, compute share
    grp["share_n"]   = grp["n_est"] / grp.groupby("REGION_int")["n_est"].transform("sum")
    grp["share_emp"] = grp["emp"]   / grp.groupby("REGION_int")["emp"].transform("sum")
    grp["share_vbp"] = grp["vbp"]   / grp.groupby("REGION_int")["vbp"].transform("sum")

    # merge MMI
    shake = pd.read_csv(DATA / "shakemap_by_region.csv")
    grp = grp.merge(shake[["enia_region", "mean_mmi"]],
                    left_on="REGION_int", right_on="enia_region", how="left")

    # ---- Top 3 sectors per region (by VBP share) ----
    print("\n=== Top 3 sectors by output share per region (pre-earthquake 2008-2009) ===\n")
    rows = []
    for ridx, rname in ENIA_REGION_NAME.items():
        sub = grp[grp["REGION_int"] == ridx].sort_values("share_vbp", ascending=False)
        mmi_str = f"MMI={sub['mean_mmi'].iloc[0]:.1f}" if (len(sub) > 0 and pd.notna(sub['mean_mmi'].iloc[0])) else "MMI=N/A"
        if len(sub) == 0:
            print(f"  Region {ridx:>2} {rname} ({mmi_str}): no ENIA data")
            continue
        top3 = sub.head(3)
        topstr = ", ".join([f"{r['sec_name']} ({r['share_vbp']*100:.0f}%)" for _, r in top3.iterrows()])
        print(f"  Region {ridx:>2} {rname:<20} ({mmi_str:>10}): {topstr}")
        for _, r in top3.iterrows():
            rows.append({
                "region_code": ridx,
                "region_name": rname,
                "mean_mmi": r["mean_mmi"],
                "sector": r["sec_name"],
                "share_n_est": r["share_n"],
                "share_emp":   r["share_emp"],
                "share_vbp":   r["share_vbp"],
                "abs_n_est":   r["n_est"],
                "abs_emp":     r["emp"],
            })
    top3_tbl = pd.DataFrame(rows)
    top3_tbl.to_csv(OUT / "industry_by_region_top3.csv", index=False)
    grp.drop(columns=["enia_region"]).to_csv(OUT / "industry_by_region_full.csv", index=False)
    print(f"\nWrote top-3 table → output/industry_by_region_top3.csv")
    print(f"Wrote full table → output/industry_by_region_full.csv")

    # ---- Stacked bar chart: composition by region, sorted by MMI ----
    # Wide pivot
    pivot = grp.pivot_table(index="REGION_int", columns="sec_name", values="share_vbp", fill_value=0)
    # only show regions with ENIA data + sort by MMI descending (most affected first)
    pivot = pivot.join(shake.set_index("enia_region")[["mean_mmi"]], how="inner")
    pivot = pivot.dropna(subset=["mean_mmi"]).sort_values("mean_mmi", ascending=False)
    region_names = [ENIA_REGION_NAME[r] + f"\nMMI={pivot.loc[r, 'mean_mmi']:.1f}" for r in pivot.index]
    sec_cols = [c for c in pivot.columns if c != "mean_mmi"]
    # collapse sectors with <3% in all regions into "Other"
    big_sec = [c for c in sec_cols if pivot[c].max() >= 0.03]
    small_sec = [c for c in sec_cols if c not in big_sec]
    if small_sec:
        pivot["Other"] = pivot[small_sec].sum(axis=1)
        plot_cols = big_sec + ["Other"]
    else:
        plot_cols = big_sec

    fig, ax = plt.subplots(figsize=(13, 7))
    bottoms = np.zeros(len(pivot))
    cmap = plt.cm.tab20
    for i, c in enumerate(plot_cols):
        ax.bar(region_names, pivot[c].values, bottom=bottoms,
               color=cmap(i % 20), label=c, edgecolor="white", linewidth=0.5)
        bottoms += pivot[c].values
    ax.set_ylabel("Share of output (VBP)")
    ax.set_xlabel("Region (sorted by MMI, highest left)")
    ax.set_title("Pre-earthquake manufacturing composition by region (2008-2009)\n"
                 "Output share by 2-digit ISIC sector")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, ncol=1)
    ax.set_ylim(0, 1.05)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_fontsize(8)
    fig.tight_layout()
    fig.savefig(FIGS / "F1_industry_composition_by_region.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIGS / 'F1_industry_composition_by_region.png'}")


if __name__ == "__main__":
    main()
