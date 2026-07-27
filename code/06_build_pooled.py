"""Build pooled cross-section parquet with harmonized industry classification.

For each year:
  - 2008-2011: each row expanded by Rev.3.1 -> Rev.4 share weights (fractional)
  - 2013-2017: each row has weight = 1.0, ind_rev4 = CIIU4

Pooled output: enia_pooled.parquet
Industry x year aggregation: enia_industry_year.parquet
Industry x region x year aggregation: enia_industry_region_year.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CROSSWALKS = ROOT / "crosswalks"

YEARS_OLD = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011]  # Rev.3 (CIIU3); pre-2008 = NUI era
YEARS_NEW = [2013, 2014, 2015, 2016, 2017]  # Rev.4 (CIIU4 field)
# (2012 dropped, 2018 dropped, 2000-2002 skipped)


def load_share_weights() -> pd.DataFrame:
    sw = pd.read_csv(CROSSWALKS / "rev31_to_rev4_share_weights.csv")
    sw["rev31"] = sw["rev31"].astype(str).str.strip()
    sw["rev4"] = sw["rev4"].astype(str).str.strip()
    return sw


def _normalize_4digit(s: pd.Series) -> pd.Series:
    """Return Series of 4-digit code strings ('1512') or NA.
    Handles float64 with .0 suffix, int, and string inputs. 2-3 digit codes
    and the '9999' sentinel are returned as NA (not full 4-digit class).
    """
    # try numeric coercion first (handles float64 '1512.0')
    num = pd.to_numeric(s, errors="coerce")
    # keep only true 4-digit codes (1000..9998); 9999 is INE "unclassified"
    valid = num.between(1000, 9998)
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    out.loc[valid] = num[valid].astype(int).astype(str)
    return out


def harmonize_year(year: int, share_weights: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"wide_{year}.parquet")
    if year in YEARS_NEW:
        df["ind_rev4"] = _normalize_4digit(df["CIIU4"])
        df["ind_weight"] = 1.0
        return df
    # YEARS_OLD path (Rev.3.1 codes)
    df["_rev31"] = _normalize_4digit(df["CIIU3"])
    merged = df.merge(
        share_weights.rename(columns={"rev31": "_rev31", "rev4": "ind_rev4", "share": "ind_weight"})[
            ["_rev31", "ind_rev4", "ind_weight"]
        ],
        on="_rev31",
        how="left",
    )
    # records with Rev.3.1 missing/sub-4-digit/unmatched -> NaN industry, weight=1
    merged.loc[merged["ind_weight"].isna(), "ind_weight"] = 1.0
    merged = merged.drop(columns=["_rev31"])
    return merged


def main():
    sw = load_share_weights()
    print(f"Loaded share weights: {len(sw)} rev31->rev4 pairs across {sw['rev31'].nunique()} rev31 codes")

    pieces = []
    for y in YEARS_OLD + YEARS_NEW:
        before = pd.read_parquet(DATA / f"wide_{y}.parquet").shape[0]
        h = harmonize_year(y, sw)
        after = h.shape[0]
        unmapped = h["ind_rev4"].isna().sum()
        expansion = after / before
        print(f"  {y}: {before} -> {after} rows (x{expansion:.2f})  unmapped industry: {unmapped}")
        pieces.append(h)

    pooled = pd.concat(pieces, ignore_index=True, sort=False)
    print(f"\nPooled shape: {pooled.shape}")
    print(f"Total weight sum: {pooled['ind_weight'].sum():,.1f} "
          f"(should equal raw establishment-year count: "
          f"{sum(pd.read_parquet(DATA / f'wide_{y}.parquet').shape[0] for y in YEARS_OLD + YEARS_NEW)})")

    # Normalize string-ish columns (CIIU3/CIIU4/etc) so pyarrow can write them.
    # The issue: a column can be str in some years and float (NaN) in years where it's absent.
    for col in pooled.columns:
        if pooled[col].dtype == "object":
            pooled[col] = pooled[col].where(pooled[col].notna(), None).astype("string")

    pooled.to_parquet(DATA / "enia_pooled.parquet", index=False)
    print(f"Wrote data/enia_pooled.parquet  ({pooled.shape})")

    # ----- Industry x year aggregation -----
    print("\nBuilding industry x year cells...")
    # Outcome columns we want (those present in any year)
    outcome_cols = ["EMPTOT", "TOTHOM", "TOTMUJ",
                    "RENIMP", "CONIMP", "TIMIMP", "PATIMP", "OTRIMP",
                    "VBP", "CI", "VA"]
    avail = [c for c in outcome_cols if c in pooled.columns]
    print(f"  Available outcomes for aggregation: {avail}")

    def weighted_agg(group: pd.DataFrame) -> pd.Series:
        w = group["ind_weight"].fillna(0)
        out = {
            "n_obs": len(group),
            "weight_sum": w.sum(),
            "n_establishments": (group["ind_weight"] > 0).sum(),
        }
        for c in avail:
            vals = pd.to_numeric(group[c], errors="coerce")
            mask = vals.notna() & w.notna()
            if mask.sum() == 0:
                out[f"{c}_wsum"] = np.nan
                out[f"{c}_wmean"] = np.nan
            else:
                out[f"{c}_wsum"] = (vals[mask] * w[mask]).sum()
                out[f"{c}_wmean"] = (vals[mask] * w[mask]).sum() / w[mask].sum()
        return pd.Series(out)

    iy = (pooled.dropna(subset=["ind_rev4", "ANIO"])
                .groupby(["ind_rev4", "ANIO"], group_keys=False)
                .apply(weighted_agg, include_groups=False)
                .reset_index())
    iy.to_parquet(DATA / "enia_industry_year.parquet", index=False)
    print(f"  Wrote data/enia_industry_year.parquet  ({iy.shape})  "
          f"({iy['ind_rev4'].nunique()} industries x {iy['ANIO'].nunique()} years)")

    # ----- Industry x region x year aggregation -----
    print("\nBuilding industry x region x year cells...")
    iry = (pooled.dropna(subset=["ind_rev4", "ANIO", "REGION"])
                 .groupby(["ind_rev4", "REGION", "ANIO"], group_keys=False)
                 .apply(weighted_agg, include_groups=False)
                 .reset_index())
    iry.to_parquet(DATA / "enia_industry_region_year.parquet", index=False)
    print(f"  Wrote data/enia_industry_region_year.parquet  ({iry.shape})  "
          f"({iry['ind_rev4'].nunique()} x {iry['REGION'].nunique()} x {iry['ANIO'].nunique()})")


if __name__ == "__main__":
    main()
