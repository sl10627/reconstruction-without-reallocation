"""Harmonize CASEN 2003 to the migration-test schema and cache a slim parquet.

CASEN 2003 has 13 regions (Los Rios / Arica were split off only in 2007) and uses
different variable names. I map:
  r (roman numerals) -> REGION 1-13  (r.m. = 13 Metropolitana)
  educ (string labels) -> skill: unskilled = none/basica; skilled = superior (CFT/IP/univ)
  activ -> in labour force (ocupado or desocupado)
  expr  -> population weight ; edad -> working age 15-64

Input:  data/_casen_extract/casen2003.dta
Output: data/casen2003_slim.parquet  (working-age persons, columns REGION, unskilled,
skilled, in_lf, w)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = DATA / "_casen_extract" / "casen2003.dta"   # extracted from data/casen2003stata.rar

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8,
         "ix": 9, "x": 10, "xi": 11, "xii": 12, "r.m.": 13}
UNSK = {"sin educacion formal", "basica incompleta", "basica completa."}
SK = {"c.f.t/i.p incompleta.", "c.f.t/i.p completa.", "universidad incompleta", "universidad completa"}
INLF = {"ocupado", "desocupado"}


def main():
    d = pd.read_stata(SRC, columns=["r", "educ", "activ", "expr", "edad"])
    d["REGION"] = d["r"].astype(str).str.strip().str.lower().map(ROMAN)
    d = d[(d["edad"] >= 15) & (d["edad"] <= 64) & d["REGION"].notna()].copy()
    ed = d["educ"].astype(str).str.strip().str.lower()
    d["unskilled"] = ed.isin(UNSK)
    d["skilled"] = ed.isin(SK)
    d["in_lf"] = d["activ"].astype(str).str.strip().str.lower().isin(INLF)
    d["w"] = pd.to_numeric(d["expr"], errors="coerce")
    out = d[["REGION", "unskilled", "skilled", "in_lf", "w"]].copy()
    out["REGION"] = out["REGION"].astype(int)
    out.to_parquet(DATA / "casen2003_slim.parquet", index=False)
    print(f"Wrote {DATA/'casen2003_slim.parquet'}  shape={out.shape}")
    print("regions:", sorted(out.REGION.unique()))
    print(f"weighted pop (working age): {out['w'].sum():,.0f}")
    print(f"unskilled share: {out.loc[out.unskilled,'w'].sum()/out['w'].sum():.3f}, "
          f"skilled share: {out.loc[out.skilled,'w'].sum()/out['w'].sum():.3f}")


if __name__ == "__main__":
    main()
