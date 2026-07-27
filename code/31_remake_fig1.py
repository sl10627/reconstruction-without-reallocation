"""Regenerate Figure 1 (central-Chile MMI zoom map) with correctly accented region
labels. Does NOT recompute MMI -- it reuses data/shakemap_by_region.csv and the GADM
region polygons, only redrawing the map. Overwrites output/figs/E2_chile_mmi_map_zoom.png.
"""
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
QDIR, DATA, FIGS = ROOT / "geodata", ROOT / "data", ROOT / "output" / "figs"

ENIA_REGION_MAP = {
    1: "Tarapacá", 2: "Antofagasta", 3: "Atacama", 4: "Coquimbo", 5: "Valparaíso",
    6: "Libertador General Bernardo O'Hi", 7: "Maule", 8: "Bío-Bío", 9: "Araucanía",
    10: "Los Lagos", 11: "Aysén del General Ibañez del Cam",
    12: "Magallanes y Antártica Chilena", 13: "Santiago Metropolitan",
    14: "Los Ríos", 15: "Arica y Parinacota", 16: "Ñuble",
}
# short, correctly accented display labels
SHORT = {
    1: "Tarapacá", 2: "Antofagasta", 3: "Atacama", 4: "Coquimbo", 5: "Valparaíso",
    6: "O'Higgins", 7: "Maule", 8: "Bío-Bío", 9: "Araucanía", 10: "Los Lagos",
    11: "Aysén", 12: "Magallanes", 13: "Santiago Metro.", 14: "Los Ríos",
    15: "Arica y Parinacota", 16: "Ñuble",
}

chile = gpd.read_file(QDIR / "gadm_chile" / "gadm41_CHL_1.shp")[["NAME_1", "geometry"]].copy()
chile["NAME_1_clean"] = chile["NAME_1"].astype(str)
reverse_map = {v: k for k, v in ENIA_REGION_MAP.items()}
chile["enia_region"] = chile["NAME_1_clean"].map(reverse_map)
unmatched = chile[chile["enia_region"].isna()]
if len(unmatched):
    print("WARNING unmatched:", list(unmatched["NAME_1_clean"]))

shake = pd.read_csv(DATA / "shakemap_by_region.csv")
chile["mean_mmi"] = chile["enia_region"].map(shake.set_index("enia_region")["mean_mmi"])
chile_geo = chile.to_crs("EPSG:4326")

fig, ax = plt.subplots(figsize=(8, 10))
chile_geo.plot(column="mean_mmi", ax=ax, cmap="YlOrRd", legend=True,
               edgecolor="black", linewidth=0.8,
               legend_kwds={"label": "Area-weighted mean MMI", "shrink": 0.6},
               missing_kwds={"color": "lightgray"})
for _, r in chile_geo.iterrows():
    reg = r["enia_region"]
    if pd.isna(reg):
        continue
    cent = r.geometry.representative_point()
    name = SHORT.get(int(reg), str(r["NAME_1_clean"])[:14])
    label = f"{name}\nMMI={r['mean_mmi']:.1f}" if not np.isnan(r["mean_mmi"]) else name
    ax.annotate(label, xy=(cent.x, cent.y), fontsize=8, ha="center", va="center")
ax.set_xlim(-77, -68)
ax.set_ylim(-40, -28)
ax.scatter([-72.898], [-36.122], s=250, marker="*", color="red",
           edgecolors="black", linewidths=1.5, zorder=10, label="Epicenter")
ax.legend(loc="upper left")
ax.set_title("Central Chile zoom: 2010 Earthquake MMI by region\n"
             "Red star = epicenter (offshore Maule)", fontsize=11)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
fig.tight_layout()
out = FIGS / "E2_chile_mmi_map_zoom.png"
fig.savefig(out, dpi=140)
plt.close(fig)
print("Wrote", out)
