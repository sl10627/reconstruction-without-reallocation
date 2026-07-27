"""Build per-year wide tables for ENIA 2008-2017 (drop 2012).

For each year:
  1. Load 3 PARTE CSVs.
  2. Apply dedup on ID (access-parser bug for 2016).
  3. Apply known field renames:
     - 2012:        CIIU       -> CIIU3
     - 2008-2013:   VBPb       -> VBP   (P2; 2008 also has 'VAB' which we ignore for now)
  4. Verify ID alignment across PARTEs (should be identical sets per year).
  5. Outer-join P1 + P2 + P3 on ID, producing wide_{year}.parquet.
  6. Tag each row with ANIO (already present), and add outcome-eligibility flags:
        emp_eligible    = year >= 2009 (employment grid stable)
        cap_eligible    = year >= 2009 (capital field stable)
        wage_eligible   = (2009 <= year <= 2016)
        vbp_eligible    = year >= 2013

Output:
  data/wide_{year}.parquet   for year in {2008,2009,2010,2011,2013,2014,2015,2016,2017}
  data/wide_manifest.csv     summary table: rows per year, columns retained, drops
"""

from __future__ import annotations

import ast
import struct
from pathlib import Path

import numpy as np
import pandas as pd


def _decode_access_bytes(x):
    """access-parser stores some 2009 numeric fields as 17-byte blobs whose
    last 4 bytes contain the value as little-endian uint32.

    The raw CSV serialization is the Python repr of the bytes object, e.g.
    "b'\\x00\\x00...\\x89t\\x18\\x00'". We literal_eval back to bytes and
    extract the trailing int32. Verified against 2008 and 2010 magnitudes
    for RENIMP, CONIMP, VBP, CBUMAQ, CBUEDI — all within 10-30% (consistent
    with INE's multiplicative noise + sampling variation).
    """
    if not isinstance(x, str):
        return x  # already numeric
    if not x.startswith("b'") or "\\x" not in x:
        return x  # not a bytes-blob
    try:
        b = ast.literal_eval(x)
        if isinstance(b, bytes) and len(b) >= 4:
            return float(struct.unpack("<I", b[-4:])[0])
    except Exception:
        pass
    return None


def _decode_bytes_columns(df: pd.DataFrame) -> int:
    """In place: detect and decode bytes-blob columns. Returns # columns fixed."""
    fixed = 0
    for col in df.columns:
        s = df[col]
        nonnull = s.dropna()
        if nonnull.empty:
            continue
        first = nonnull.iloc[0]
        if isinstance(first, str) and first.startswith("b'") and "\\x" in first:
            df[col] = s.map(_decode_access_bytes)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            fixed += 1
    return fixed

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# pre-2008 (NUI panel era, CIIU Rev.3) added to extend the pre-trend window for
# the 2010-earthquake design. 2006 retrieved from INE (released, 1995+ series);
# 2000-2002 (older vintage) skipped.
YEARS = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016, 2017]  # drop 2012


def load_parte(year: int, parte: int) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"{year}_PARTE{parte}.csv", low_memory=False)
    # pre-2008 files key on NUI (stable panel id) instead of ID -> normalize
    if "ID" not in df.columns and "NUI" in df.columns:
        df.rename(columns={"NUI": "ID"}, inplace=True)
    # dedup on ID (access-parser bug for 2016; harmless elsewhere)
    before = len(df)
    df = df.drop_duplicates(subset=["ID"], keep="first").reset_index(drop=True)
    if before != len(df):
        print(f"  {year} P{parte}: dropped {before - len(df)} duplicate-ID rows ({before} -> {len(df)})")
    # decode bytes-blob columns (access-parser bug, observed in 2009)
    n_fixed = _decode_bytes_columns(df)
    if n_fixed:
        print(f"  {year} P{parte}: decoded {n_fixed} bytes-blob columns to numeric")
    return df


def harmonize_inplace(df: pd.DataFrame, year: int, parte: int) -> None:
    """Rename columns to bring them into a common namespace."""
    renames = {}

    # 2012 P1: CIIU (no digit) -> CIIU3
    if year == 2012 and parte == 1 and "CIIU" in df.columns and "CIIU3" not in df.columns:
        renames["CIIU"] = "CIIU3"

    # VBPb (2008-2013, P2) -> VBP. NB: 2008 P2 is 'VBPB' uppercase.
    if parte == 2:
        for old in ["VBPb", "VBPB"]:
            if old in df.columns and "VBP" not in df.columns:
                renames[old] = "VBP"
                break

    if renames:
        df.rename(columns=renames, inplace=True)
        print(f"  {year} P{parte}: renamed {renames}")


def build_wide_year(year: int) -> dict:
    print(f"\n=== {year} ===")
    parts = {}
    for p in [1, 2, 3]:
        df = load_parte(year, p)
        harmonize_inplace(df, year, p)
        parts[p] = df

    # check ID set alignment
    ids = {p: set(parts[p]["ID"].dropna()) for p in [1, 2, 3]}
    all_ids = ids[1] | ids[2] | ids[3]
    common = ids[1] & ids[2] & ids[3]
    print(f"  ID sets: P1={len(ids[1])}, P2={len(ids[2])}, P3={len(ids[3])}, "
          f"union={len(all_ids)}, intersection={len(common)}")
    if len(all_ids) != len(common):
        print(f"  WARNING: {len(all_ids) - len(common)} IDs not in all 3 partes")

    # join: outer on ID so we don't drop establishments only in 2 partes
    # but drop the duplicated metadata columns coming from P2/P3 (ANIO etc.)
    # strategy: take P1 as base, then merge P2 and P3 with non-key cols only
    base = parts[1].copy()
    for p in [2, 3]:
        other = parts[p].copy()
        # drop columns that already exist in base, except ID (the join key)
        overlap = [c for c in other.columns if c in base.columns and c != "ID"]
        if overlap:
            other = other.drop(columns=overlap)
        base = base.merge(other, on="ID", how="outer")

    # add outcome-eligibility flags
    base["emp_eligible"] = year >= 2009
    base["cap_eligible"] = year >= 2009
    base["wage_eligible"] = (2009 <= year <= 2016)
    base["vbp_eligible"] = year >= 2013

    # ensure ANIO column present and consistent
    if "ANIO" not in base.columns:
        base["ANIO"] = year
    else:
        # if mismatched, warn
        bad = base["ANIO"].dropna().unique()
        if len(bad) != 1 or int(bad[0]) != year:
            print(f"  WARNING: ANIO values for {year} are {bad}")

    out_path = DATA / f"wide_{year}.parquet"
    base.to_parquet(out_path, index=False)
    print(f"  wrote {out_path.name}  rows={len(base)}  cols={base.shape[1]}")

    return {
        "year": year,
        "n_rows": len(base),
        "n_cols": base.shape[1],
        "n_ids_union": len(all_ids),
        "n_ids_common": len(common),
        "emp_eligible": year >= 2009,
        "cap_eligible": year >= 2009,
        "wage_eligible": (2009 <= year <= 2016),
        "vbp_eligible": year >= 2013,
    }


def main():
    manifest = []
    for y in YEARS:
        manifest.append(build_wide_year(y))
    mf = pd.DataFrame(manifest)
    mf.to_csv(DATA / "wide_manifest.csv", index=False)
    print("\n=== Manifest ===")
    print(mf.to_string(index=False))


if __name__ == "__main__":
    main()
