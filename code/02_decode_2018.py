"""Decode anonymized 2018 ENIA field names by matching distribution signatures
to 2017 (last year with semantic names).

Strategy:
1. Compute a distribution signature for each numeric field:
   (n_unique, frac_zero, frac_missing, mean, std, q05, q25, q50, q75, q95, min, max)
2. For each anonymized 2018 field, find nearest neighbor in 2017 by signature.
3. Validate methodology on 2016->2017 (both have known names): the matched
   2017 name for each 2016 field should equal the 2016 name itself when both
   exist, giving an accuracy score.

Notes:
- We pool across all 3 PARTEs since INE reorganized partes between 2017 and 2018.
- Apply dedup_on_id first (2016 / 2018 access-parser bug).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"


def load_year(year: int) -> dict[int, pd.DataFrame]:
    out = {}
    for p in [1, 2, 3]:
        df = pd.read_csv(RAW / f"{year}_PARTE{p}.csv", low_memory=False)
        df = df.drop_duplicates(subset=["ID"], keep="first").reset_index(drop=True)
        out[p] = df
    return out


def signature(series: pd.Series) -> np.ndarray:
    """Return distribution signature for a numeric column.

    Non-numeric columns return None.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return None
    n = len(s)
    nv = s.dropna()
    if nv.empty:
        return None
    return np.array([
        nv.nunique() / max(n, 1),
        (s == 0).mean(),
        s.isna().mean(),
        nv.mean(),
        nv.std() if len(nv) > 1 else 0.0,
        nv.quantile(0.05),
        nv.quantile(0.25),
        nv.quantile(0.50),
        nv.quantile(0.75),
        nv.quantile(0.95),
        nv.min(),
        nv.max(),
    ], dtype=float)


def normalize_for_dist(sig: np.ndarray) -> np.ndarray:
    """Log-scale the location/scale components so values across orders of
    magnitude are comparable.
    """
    out = sig.copy()
    for i in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
        out[i] = np.sign(out[i]) * np.log1p(np.abs(out[i]))
    return out


def all_field_sigs(year_data: dict[int, pd.DataFrame], skip_cols: set[str]) -> dict[str, np.ndarray]:
    sigs = {}
    for p, df in year_data.items():
        for col in df.columns:
            if col in skip_cols:
                continue
            sig = signature(df[col])
            if sig is None:
                continue
            sigs[col] = normalize_for_dist(sig)
    return sigs


def nearest_neighbor(query_sig: np.ndarray, candidates: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    dists = []
    for name, sig in candidates.items():
        d = np.linalg.norm(query_sig - sig)
        dists.append((name, d))
    dists.sort(key=lambda x: x[1])
    return dists


SKIP = {"ID", "ANIO", "IMPUTACION", "REGION", "TAMANO", "CIIU4", "CIIU3", "CIIU004", "CLASIFICACION", "TEED"}


def validate_on_known_pair(year_source: int, year_target: int) -> tuple[float, pd.DataFrame]:
    """Run the matching algorithm on a (source, target) pair where both
    have known names. Returns top-1 accuracy and per-field results.
    """
    src = load_year(year_source)
    tgt = load_year(year_target)
    src_sigs = all_field_sigs(src, SKIP)
    tgt_sigs = all_field_sigs(tgt, SKIP)
    common = set(src_sigs) & set(tgt_sigs)
    rows = []
    correct_top1 = 0
    correct_top5 = 0
    for col in sorted(common):
        nn = nearest_neighbor(src_sigs[col], tgt_sigs)
        top1 = nn[0][0]
        top5 = [x[0] for x in nn[:5]]
        if top1 == col:
            correct_top1 += 1
        if col in top5:
            correct_top5 += 1
        rows.append({
            "src_col": col,
            "top1_match": top1,
            "top1_dist": nn[0][1],
            "top2_match": nn[1][0] if len(nn) > 1 else "",
            "top2_dist": nn[1][1] if len(nn) > 1 else np.nan,
            "in_top5": col in top5,
            "is_top1": top1 == col,
        })
    df_out = pd.DataFrame(rows)
    return correct_top1 / max(len(common), 1), correct_top5 / max(len(common), 1), df_out


def main():
    # --- Validation: 2016 -> 2017 ---
    print("=== Validation: match 2016 fields to 2017 candidates ===\n")
    acc1, acc5, df_val = validate_on_known_pair(2016, 2017)
    print(f"Common fields: {len(df_val)}")
    print(f"Top-1 accuracy: {acc1:.1%}")
    print(f"Top-5 accuracy: {acc5:.1%}\n")

    # Inspect some failures
    fails = df_val[~df_val["is_top1"]].sort_values("top1_dist")
    print(f"Failures: {len(fails)} fields where top-1 != true name")
    print(fails.head(15).to_string(index=False))

    df_val.to_csv(ROOT / "output" / "decode_validation_2016_to_2017.csv", index=False)
    print(f"\nFull validation table -> output/decode_validation_2016_to_2017.csv")

    # --- Apply: 2018 anon fields -> 2017 candidates ---
    print("\n\n=== Apply: match 2018 anonymized fields to 2017 candidates ===\n")
    d2018 = load_year(2018)
    d2017 = load_year(2017)
    src_sigs = all_field_sigs(d2018, SKIP)
    tgt_sigs = all_field_sigs(d2017, SKIP)
    rows = []
    for col, sig in src_sigs.items():
        nn = nearest_neighbor(sig, tgt_sigs)
        rows.append({
            "anon_2018": col,
            "top1": nn[0][0], "d1": nn[0][1],
            "top2": nn[1][0] if len(nn) > 1 else "", "d2": nn[1][1] if len(nn) > 1 else np.nan,
            "top3": nn[2][0] if len(nn) > 2 else "", "d3": nn[2][1] if len(nn) > 2 else np.nan,
        })
    df_apply = pd.DataFrame(rows).sort_values("anon_2018")
    df_apply.to_csv(ROOT / "output" / "decode_2018_candidates.csv", index=False)
    print(f"{len(df_apply)} 2018 anon fields mapped. -> output/decode_2018_candidates.csv\n")
    print(df_apply.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
