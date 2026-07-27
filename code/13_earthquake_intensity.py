"""Compute area-weighted MMI by Chile region and produce paper Figure 1 map.

Inputs (relative to project root, see QDIR below):
  geodata/gadm_chile/gadm41_CHL_1.shp
  geodata/shakemap_shape/mi.shp
  geodata/shakemap_shape/pga.shp

Outputs:
  data/shakemap_by_region.csv
  output/figs/E1_chile_mmi_map.png
  output/figs/E2_chile_mmi_map_with_ENIA.png   (zoomed central Chile)
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
FIGS = OUT / "figs"
QDIR = ROOT / "geodata"   # earthquake shapefiles (GADM regions + USGS ShakeMap)

# ENIA REGION code (1-15) → GADM NAME_1
ENIA_REGION_MAP = {
    1:  "Tarapacá",
    2:  "Antofagasta",
    3:  "Atacama",
    4:  "Coquimbo",
    5:  "Valparaíso",
    6:  "Libertador General Bernardo O'Hi",   # truncated in GADM
    7:  "Maule",
    8:  "Bío-Bío",
    9:  "Araucanía",
    10: "Los Lagos",
    11: "Aysén del General Ibañez del Cam",   # truncated in GADM
    12: "Magallanes y Antártica Chilena",
    13: "Santiago Metropolitan",
    14: "Los Ríos",
    15: "Arica y Parinacota",
    16: "Ñuble",  # split from Bío-Bío in 2018, not in ENIA 2010-era
}


def main():
    print("Loading Chile region shapefile...")
    chile = gpd.read_file(QDIR / "gadm_chile" / "gadm41_CHL_1.shp")
    chile = chile[["NAME_1", "GID_1", "geometry"]].copy()
    chile["NAME_1_clean"] = chile["NAME_1"].astype(str)
    print(f"  {len(chile)} regions, CRS = {chile.crs}")

    # Map ENIA REGION code to GADM NAME_1
    reverse_map = {v: k for k, v in ENIA_REGION_MAP.items()}
    chile["enia_region"] = chile["NAME_1_clean"].map(reverse_map)
    unmatched = chile[chile["enia_region"].isna()]
    if len(unmatched):
        print(f"  WARNING: {len(unmatched)} regions unmatched: {list(unmatched['NAME_1_clean'])}")
    else:
        print("  All 16 regions matched to ENIA codes")

    print("\nLoading ShakeMap MMI polygons...")
    mmi = gpd.read_file(QDIR / "shakemap_shape" / "mi.shp")
    mmi = mmi[["PARAMVALUE", "geometry"]].copy()
    mmi.rename(columns={"PARAMVALUE": "mmi"}, inplace=True)
    print(f"  {len(mmi)} MMI polygons, CRS = {mmi.crs}")
    print(f"  MMI value range: {mmi['mmi'].min():.1f} to {mmi['mmi'].max():.1f}")

    # reproject both to a planar projection for accurate area math
    # Chile-friendly: SIRGAS 2000 / UTM zone 19S (EPSG:32719) - rough OK
    PLANAR_CRS = "EPSG:32719"
    chile_planar = chile.to_crs(PLANAR_CRS)
    mmi_planar = mmi.to_crs(PLANAR_CRS)

    print("\nComputing area-weighted MMI per region (spatial intersection)...")
    rows = []
    for _, region_row in chile_planar.iterrows():
        region_geom = region_row.geometry
        region_area = region_geom.area
        # intersect with all MMI polygons
        clipped = mmi_planar.copy()
        clipped["intersect"] = mmi_planar.geometry.intersection(region_geom)
        clipped["i_area"] = clipped["intersect"].area
        clipped = clipped[clipped["i_area"] > 0]
        if len(clipped) == 0:
            mean_mmi = np.nan
            max_mmi = np.nan
            area_above_7 = 0
        else:
            mean_mmi = (clipped["mmi"] * clipped["i_area"]).sum() / clipped["i_area"].sum()
            max_mmi = clipped["mmi"].max()
            area_above_7 = clipped.loc[clipped["mmi"] >= 7, "i_area"].sum() / region_area
        rows.append({
            "enia_region": region_row["enia_region"],
            "region_name": region_row["NAME_1"],
            "mean_mmi":    mean_mmi,
            "max_mmi":     max_mmi,
            "frac_area_mmi_ge_7": area_above_7,
            "region_area_m2":   region_area,
        })

    shake = pd.DataFrame(rows).sort_values("enia_region")
    print("\n=== Area-weighted MMI per Chile region ===")
    print(shake.round(2).to_string(index=False))
    shake.to_csv(DATA / "shakemap_by_region.csv", index=False)
    print(f"\nWrote {DATA / 'shakemap_by_region.csv'}")

    # ===== Make Figure 1: Chile map colored by mean MMI =====
    chile["mean_mmi"] = chile["enia_region"].map(shake.set_index("enia_region")["mean_mmi"])
    # Re-project chile back to lat/lon for nicer plot
    chile_geo = chile.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(7, 12))
    chile_geo.plot(column="mean_mmi", ax=ax, cmap="YlOrRd",
                   legend=True, edgecolor="black", linewidth=0.5,
                   legend_kwds={"label": "Area-weighted mean MMI", "shrink": 0.5},
                   missing_kwds={"color": "lightgray", "label": "No MMI"})
    # Annotate regions with name + MMI
    for _, r in chile_geo.iterrows():
        centroid = r.geometry.representative_point()
        label = f"{r['NAME_1'][:12]}\n({r['mean_mmi']:.1f})" if not np.isnan(r['mean_mmi']) else r['NAME_1'][:12]
        ax.annotate(label, xy=(centroid.x, centroid.y),
                    fontsize=6, ha="center", va="center", color="black")
    ax.set_title("2010 Chile Earthquake (Feb 27, Mw 8.8): Area-weighted MMI by Region\n"
                 "Source: USGS ShakeMap; region polygons from GADM 4.1",
                 fontsize=11)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(FIGS / "E1_chile_mmi_map.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {FIGS / 'E1_chile_mmi_map.png'}")

    # ===== Make Figure 2: Zoomed central Chile (the affected zones) =====
    fig, ax = plt.subplots(figsize=(8, 10))
    chile_geo.plot(column="mean_mmi", ax=ax, cmap="YlOrRd",
                   legend=True, edgecolor="black", linewidth=0.8,
                   legend_kwds={"label": "Area-weighted mean MMI", "shrink": 0.6},
                   missing_kwds={"color": "lightgray"})
    for _, r in chile_geo.iterrows():
        cent = r.geometry.representative_point()
        label = f"{r['NAME_1'][:14]}\nMMI={r['mean_mmi']:.1f}" if not np.isnan(r['mean_mmi']) else r['NAME_1'][:14]
        ax.annotate(label, xy=(cent.x, cent.y), fontsize=8, ha="center", va="center")
    # zoom to central Chile (where earthquake hit)
    ax.set_xlim(-77, -68); ax.set_ylim(-40, -28)
    # mark epicenter (USGS): -36.122 lat, -72.898 lon
    ax.scatter([-72.898], [-36.122], s=250, marker="*", color="red",
               edgecolors="black", linewidths=1.5, zorder=10, label="Epicenter")
    ax.legend(loc="upper left")
    ax.set_title("Central Chile zoom: 2010 Earthquake MMI by region\n"
                 "Red star = epicenter (offshore Maule)",
                 fontsize=11)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(FIGS / "E2_chile_mmi_map_zoom.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {FIGS / 'E2_chile_mmi_map_zoom.png'}")


if __name__ == "__main__":
    main()
