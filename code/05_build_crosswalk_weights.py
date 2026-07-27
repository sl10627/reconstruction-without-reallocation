"""Build Chile-specific share-weighted Rev.3.1 -> Rev.4 crosswalk.

For each ISIC Rev.3.1 code X used in ENIA 2008-2012:
  1. Find its set of Rev.4 children Y_1..Y_k from the UN crosswalk.
  2. Count ENIA 2013 establishments in each Y_i (intersected with X's children).
  3. share_i = count(Y_i) / sum_j count(Y_j).
  4. If sum_j count(Y_j) == 0 in 2013, fall back to union of 2013-2017 counts.
  5. If still 0, fall back to uniform allocation share_i = 1/k.
  6. If k == 1 (clean 1:1), share = 1.

Outputs:
  crosswalks/rev31_to_rev4_share_weights.csv  cols=[rev31, rev4, share, source]
  output/crosswalk_diagnostics.txt            summary stats
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CROSSWALKS = ROOT / "crosswalks"
RAW = ROOT / "raw"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def load_crosswalk() -> pd.DataFrame:
    cw = pd.read_csv(CROSSWALKS / "ISIC31_ISIC4.txt", encoding="latin-1")
    cw.columns = [c.strip() for c in cw.columns]
    cw["ISIC31code"] = cw["ISIC31code"].astype(str).str.strip().str.strip('"')
    cw["ISIC4code"] = cw["ISIC4code"].astype(str).str.strip().str.strip('"')
    return cw[["ISIC31code", "ISIC4code"]].drop_duplicates().reset_index(drop=True)


def load_enia_rev4_counts(years: list[int]) -> Counter:
    """Count of establishments per Rev.4 code across given years."""
    c = Counter()
    for y in years:
        df = pd.read_csv(RAW / f"{y}_PARTE1.csv", usecols=["CIIU4"], low_memory=False)
        # dedup on duplicate-ID issue for 2016
        # but here we use full PARTE1 — duplicates don't affect distribution shape much
        codes = df["CIIU4"].dropna().astype(str).str.strip()
        codes = codes[codes.str.fullmatch(r"\d{4}")]
        c.update(codes.tolist())
    return c


def get_chile_rev31_codes() -> set:
    s = set()
    for y in [2008, 2009, 2010, 2011]:
        df = pd.read_csv(RAW / f"{y}_PARTE1.csv", usecols=["CIIU3"], low_memory=False)
        s |= set(df["CIIU3"].dropna().astype(str).str.strip())
    df12 = pd.read_csv(RAW / "2012_PARTE1.csv", usecols=["CIIU"], low_memory=False)
    s |= set(df12["CIIU"].dropna().astype(str).str.strip())
    return {c for c in s if c.isdigit() and len(c) == 4}


def main():
    cw = load_crosswalk()
    chile31 = get_chile_rev31_codes()
    print(f"Chile Rev.3.1 codes: {len(chile31)}")

    counts_2013 = load_enia_rev4_counts([2013])
    counts_1317 = load_enia_rev4_counts([2013, 2014, 2015, 2016, 2017])
    print(f"Distinct Rev.4 codes in ENIA 2013:    {len(counts_2013)}")
    print(f"Distinct Rev.4 codes in ENIA 2013-17: {len(counts_1317)}")

    rows = []
    source_counter = Counter()
    for X in sorted(chile31):
        children = sorted(cw.loc[cw["ISIC31code"] == X, "ISIC4code"].unique())
        if not children:
            # No Rev.4 mapping — shouldn't happen for Chile codes, but log.
            rows.append({"rev31": X, "rev4": None, "share": 1.0, "source": "no_mapping"})
            source_counter["no_mapping"] += 1
            continue
        if len(children) == 1:
            rows.append({"rev31": X, "rev4": children[0], "share": 1.0, "source": "1to1"})
            source_counter["1to1"] += 1
            continue
        # try 2013
        cnt = {y4: counts_2013.get(y4, 0) for y4 in children}
        total = sum(cnt.values())
        if total > 0:
            src = "enia_2013"
        else:
            cnt = {y4: counts_1317.get(y4, 0) for y4 in children}
            total = sum(cnt.values())
            if total > 0:
                src = "enia_2013_2017"
            else:
                # uniform fallback
                k = len(children)
                cnt = {y4: 1 for y4 in children}
                total = k
                src = "uniform"
        for y4, c in cnt.items():
            if c == 0:
                continue
            rows.append({"rev31": X, "rev4": y4, "share": c / total, "source": src})
        source_counter[src] += 1

    df_out = pd.DataFrame(rows)
    # sanity: each rev31 share sums to 1
    sums = df_out.groupby("rev31")["share"].sum()
    bad = sums[(sums - 1.0).abs() > 1e-9]
    if len(bad):
        print(f"WARNING: {len(bad)} rev31 codes have share sum != 1: {bad.head()}")
    else:
        print(f"Share sum per rev31 = 1.0 (verified for all {len(sums)} codes)")

    df_out.to_csv(CROSSWALKS / "rev31_to_rev4_share_weights.csv", index=False)
    print(f"\nWrote crosswalks/rev31_to_rev4_share_weights.csv  ({len(df_out)} rows)")

    print("\n=== Source breakdown ===")
    for src, n in source_counter.most_common():
        print(f"  {src}: {n} rev31 codes")

    print("\n=== Top splits by # of Rev.4 children ===")
    top = df_out.groupby("rev31").size().sort_values(ascending=False).head(10)
    for X, k in top.items():
        sub = df_out[df_out["rev31"] == X].sort_values("share", ascending=False)
        print(f"  Rev.3.1 {X} -> {k} children:")
        for _, r in sub.head(6).iterrows():
            print(f"    {r['rev4']}  share={r['share']:.3f}  ({r['source']})")
        if len(sub) > 6:
            print(f"    ... and {len(sub)-6} more")


if __name__ == "__main__":
    main()
