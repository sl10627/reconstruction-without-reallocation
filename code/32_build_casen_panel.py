"""Build the harmonized six-wave CASEN person panel used by the CASEN checks.

CASEN waves 2006, 2009 (pre-earthquake) and 2011, 2013, 2015, 2017 (post). Each wave
names its income, industry and occupational-category variables differently, and VARMAP
maps them to one schema. The panel keeps every person record, and the downstream scripts
(25, 27, 28) apply their own age, labour-force and industry filters.

Wages: wage_raw_nominal is the sum of the monthly labour-income components plus the
annual bonuses divided by 12. wage_raw_real2017 deflates it with an approximate CPI.
The CPI values are not checked against the official BCCh series, but every
regression that uses them (27, 28) is in logs with year fixed effects, so a
year-specific deflator is absorbed by the year effects and does not move any
estimate.

Regions: 2017 introduces Nuble (16), which I merge back into Biobio (8) in region15.

Input:  data/casen_{2006,2009,2011,2013,2015,2017}.dta  (INE / Ministerio de
        Desarrollo Social public releases)
Output: data/casen_panel.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

YEARS = [2006, 2009, 2011, 2013, 2015, 2017]
CPI = {2006: 65.7, 2009: 76.9, 2011: 81.7, 2013: 87.4, 2015: 95.3, 2017: 99.4}
CPI_BASE = CPI[2017]

_REV3_INCOME = {"ann_bonus": "y4a", "ann_grat": "y4b", "ann_13th": "y4c", "ann_other": "y4d",
                "secondary": "y6", "rama": "rama1_rev3", "rama_classif": "rev3", "cat_ocup": "o15"}
VARMAP = {
    2006: {"base": "y1", "ext_overtime": "y3_1m", "ext_commission": "y3_2m",
           "ext_tips": "y3_8m", "ext_allow": "y3_6m", "ext_viatic": "y3_5m",
           "ext_other": "y3_9m", "secondary": "y6",
           "ann_bonus": "y4_1", "ann_grat": "y4_2", "ann_13th": "y4_3", "ann_other": "y4_4",
           "rama": "rama1_rev2", "rama_classif": "rev2", "cat_ocup": "o19"},
    2009: {"base": "y1", "ext_overtime": "y3m1", "ext_commission": "y3m2",
           "ext_tips": "y3m8", "ext_allow": "y3m6", "ext_viatic": "y3m5",
           "ext_other": "y3m9", "secondary": "y6",
           "ann_bonus": "y4m1", "ann_grat": "y4m2", "ann_13th": "y4m3", "ann_other": "y4m4",
           "rama": "rama1_rev2", "rama_classif": "rev2", "cat_ocup": "o23"},
    2011: {"base": "y1a", "ext_overtime": "y3am", "ext_commission": "y3bm",
           "ext_tips": "y3cm", "ext_allow": "y3dm", "ext_viatic": "y3em",
           "ext_other": "y3fm", **_REV3_INCOME},
    2013: {"base": "y1a", "ext_overtime": "y3am", "ext_commission": "y3bm",
           "ext_tips": "y3cm", "ext_allow": "y3dm", "ext_viatic": "y3em",
           "ext_other": "y3fm", **_REV3_INCOME},
    2015: {"base": "y1", "ext_overtime": "y3a", "ext_commission": "y3b",
           "ext_tips": "y3c", "ext_allow": "y3d", "ext_viatic": "y3e",
           "ext_other": "y3f", **_REV3_INCOME},
    2017: {"base": "y1", "ext_overtime": "y3a", "ext_commission": "y3b",
           "ext_tips": "y3c", "ext_allow": "y3d", "ext_viatic": "y3e",
           "ext_other": "y3f", **_REV3_INCOME},
}
WAGE_MONTHLY = ["base", "ext_overtime", "ext_commission", "ext_tips",
                "ext_allow", "ext_viatic", "ext_other", "secondary"]
WAGE_ANNUAL = ["ann_bonus", "ann_grat", "ann_13th", "ann_other"]

TREATED_REFINED = {10, 11, 14}
TREATED_BUNDLED_REV3 = {12, 13, 14, 15}
TREATED_BUNDLED_REV2 = {9}
LOWSKILL_REV3 = {1, 6}
LOWSKILL_REV2 = {1, 5}


def build_wave(year: int) -> pd.DataFrame:
    m = VARMAP[year]
    needed = ["educc", "esc", "activ", "edad", "expr", "region", "ytotcor",
              m["rama"], m["cat_ocup"]]
    needed += [m[k] for k in WAGE_MONTHLY + WAGE_ANNUAL]
    needed = list(dict.fromkeys(needed))
    df = pd.read_stata(DATA / f"casen_{year}.dta", columns=needed, convert_categoricals=False)

    out = pd.DataFrame({"year": year}, index=df.index)
    for v in ["edad", "esc", "expr", "activ", "educc", "region", "ytotcor"]:
        out[v] = df[v].values

    # Wage components: monthly items plus annual items spread over 12 months.
    monthly = sum(pd.to_numeric(df[m[k]], errors="coerce").fillna(0) for k in WAGE_MONTHLY)
    annual = sum(pd.to_numeric(df[m[k]], errors="coerce").fillna(0) for k in WAGE_ANNUAL)
    out["wage_raw_nominal"] = monthly + annual / 12.0
    out["wage_raw_real2017"] = out["wage_raw_nominal"] * (CPI_BASE / CPI[year])

    # Industry. 2006 and 2009 use the Rev. 2 one-digit classification, later waves Rev. 3.
    rama_int = pd.to_numeric(df[m["rama"]], errors="coerce")
    out["rama_raw"] = rama_int
    out["rama_classif"] = m["rama_classif"]
    if m["rama_classif"] == "rev3":
        out["treated_industry_refined"] = rama_int.isin(TREATED_REFINED).astype("Int8")
        out["treated_industry_bundled"] = rama_int.isin(TREATED_BUNDLED_REV3).astype("Int8")
        out["low_skill_industry"] = rama_int.isin(LOWSKILL_REV3).astype("Int8")
    else:
        out["treated_industry_refined"] = pd.array([pd.NA] * len(out), dtype="Int8")
        out["treated_industry_bundled"] = rama_int.isin(TREATED_BUNDLED_REV2).astype("Int8")
        out["low_skill_industry"] = rama_int.isin(LOWSKILL_REV2).astype("Int8")

    cat = pd.to_numeric(df[m["cat_ocup"]], errors="coerce")
    out["cat_ocup"] = cat
    out["private_employee"] = (cat == 5).astype("Int8")

    out["advanced_ed"] = (out["educc"] == 6).astype("Int8")
    out["advanced_ed_wb"] = out["educc"].isin([5, 6]).astype("Int8")

    out["region15"] = out["region"].replace({16: 8}) if year == 2017 else out["region"]
    out["in_main_sample"] = ((out.activ == 1) & (out.edad >= 15) &
                             (out.educc != 99) & out.educc.notna()).astype("Int8")
    return out.reset_index(drop=True)


def main():
    frames = []
    for year in YEARS:
        w = build_wave(year)
        print(f"{year}: n={len(w):,}  employed (main sample)={int(w.in_main_sample.sum()):,}")
        frames.append(w)
    panel = pd.concat(frames, ignore_index=True)

    assert panel["year"].notna().all()
    assert set(panel["year"].unique()) == set(YEARS)
    assert panel["region15"].between(1, 15).all(), "region15 should have 15 regions after the Nuble merge"

    panel.to_parquet(DATA / "casen_panel.parquet", index=False)
    print(f"Wrote {DATA / 'casen_panel.parquet'}  shape={panel.shape}")


if __name__ == "__main__":
    main()
