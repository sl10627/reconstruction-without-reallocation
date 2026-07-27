"""Build cross-year field-existence matrix for ENIA 2008-2017 and pin down
the two schema breaks:

  Break 1 (2013): CIIU3 -> CIIU4 (industry classification)
  Break 2 (2015): form reorganization (some PARTE3 fields migrate to PARTE1)

Output:
  output/field_inventory.csv        — wide table, rows = field name, cols = each (year, PARTE), value = '1' if present
  output/schema_break_2013.txt      — names that disappear/appear at 2013
  output/schema_break_2015.txt      — names that move between PARTEs or disappear/appear at 2015
  output/field_movement_2015.csv    — fields that exist both before and after 2015 but in a different PARTE
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

YEARS = list(range(2008, 2018))  # 2008..2017 inclusive; 2018 dropped


def field_list(year: int, parte: int) -> list[str]:
    df = pd.read_csv(RAW / f"{year}_PARTE{parte}.csv", nrows=1, low_memory=False)
    return list(df.columns)


def build_inventory() -> pd.DataFrame:
    """rows = field name, cols = '{year}_P{parte}', cell = 1 if present."""
    records = {}
    for y in YEARS:
        for p in [1, 2, 3]:
            for col in field_list(y, p):
                records.setdefault(col, set()).add((y, p))
    rows = []
    for field, locs in records.items():
        row = {"field": field}
        for y in YEARS:
            for p in [1, 2, 3]:
                row[f"{y}_P{p}"] = 1 if (y, p) in locs else 0
            row[f"{y}_in"] = sum(row[f"{y}_P{p}"] for p in [1, 2, 3])
        row["years_present"] = sum(1 for y in YEARS if row[f"{y}_in"] > 0)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("field").sort_index()
    return df


def detect_break_2013(inv: pd.DataFrame) -> dict:
    """Fields that exist in 2008-2012 but disappear in 2013-2017, and vice versa."""
    pre = ["2008_in", "2009_in", "2010_in", "2011_in", "2012_in"]
    post = ["2013_in", "2014_in", "2015_in", "2016_in", "2017_in"]
    pre_only = inv[(inv[pre].sum(axis=1) > 0) & (inv[post].sum(axis=1) == 0)]
    post_only = inv[(inv[pre].sum(axis=1) == 0) & (inv[post].sum(axis=1) > 0)]
    return {"pre_only": list(pre_only.index), "post_only": list(post_only.index)}


def detect_break_2015(inv: pd.DataFrame) -> dict:
    """Fields whose PARTE assignment changes between 2014 (old schema)
    and 2015 (new schema). Also fields that disappear or appear at 2015.
    """
    def parte_in_year(field: str, year: int) -> int | None:
        for p in [1, 2, 3]:
            if inv.at[field, f"{year}_P{p}"]:
                return p
        return None

    movements = []
    disappear = []
    appear = []
    for f in inv.index:
        p2014 = parte_in_year(f, 2014)
        p2015 = parte_in_year(f, 2015)
        if p2014 is not None and p2015 is None:
            disappear.append(f)
        elif p2014 is None and p2015 is not None:
            appear.append(f)
        elif p2014 is not None and p2015 is not None and p2014 != p2015:
            movements.append({"field": f, "parte_2014": p2014, "parte_2015": p2015})
    return {
        "moved": pd.DataFrame(movements),
        "disappear_at_2015": disappear,
        "appear_at_2015": appear,
    }


def main():
    inv = build_inventory()
    print(f"Total distinct fields across 2008-2017: {len(inv)}")
    inv.to_csv(OUT / "field_inventory.csv")
    print(f"  -> output/field_inventory.csv  ({inv.shape})")

    # how many fields are present every year (the stable core)
    full_coverage = inv[inv["years_present"] == 10]
    print(f"\nFields present in all 10 years: {len(full_coverage)}")
    if len(full_coverage) > 0:
        print(f"  e.g. {list(full_coverage.index[:20])}")

    # break 2013
    print("\n=== Break 1: 2013 (CIIU3 -> CIIU4) ===")
    b13 = detect_break_2013(inv)
    print(f"  pre-only (in 2008-2012 only, gone 2013+): {len(b13['pre_only'])}")
    print(f"    {b13['pre_only'][:30]}")
    print(f"  post-only (new in 2013+, absent before):  {len(b13['post_only'])}")
    print(f"    {b13['post_only'][:30]}")
    with (OUT / "schema_break_2013.txt").open("w") as f:
        f.write("=== Fields only in 2008-2012 (disappear at 2013) ===\n")
        f.write("\n".join(b13["pre_only"]))
        f.write("\n\n=== Fields only in 2013-2017 (appear at 2013) ===\n")
        f.write("\n".join(b13["post_only"]))

    # break 2015
    print("\n=== Break 2: 2015 (form reorganization) ===")
    b15 = detect_break_2015(inv)
    print(f"  fields that moved between PARTEs:        {len(b15['moved'])}")
    print(f"  fields disappear at 2015:                 {len(b15['disappear_at_2015'])}")
    print(f"  fields appear at 2015:                    {len(b15['appear_at_2015'])}")
    if len(b15["moved"]) > 0:
        movement_counts = b15["moved"].groupby(["parte_2014", "parte_2015"]).size()
        print(f"\n  PARTE migrations:")
        for (p_old, p_new), n in movement_counts.items():
            print(f"    P{p_old} (2014) -> P{p_new} (2015): {n} fields")
    b15["moved"].to_csv(OUT / "field_movement_2015.csv", index=False)
    with (OUT / "schema_break_2015.txt").open("w") as f:
        f.write("=== Fields that moved between PARTEs (2014 -> 2015) ===\n")
        f.write(b15["moved"].to_string(index=False))
        f.write("\n\n=== Disappear at 2015 ===\n")
        f.write("\n".join(b15["disappear_at_2015"]))
        f.write("\n\n=== Appear at 2015 ===\n")
        f.write("\n".join(b15["appear_at_2015"]))

    # paper-critical fields: check their presence
    critical = [
        "ID", "REGION", "ANIO", "TAMANO",
        "CIIU3", "CIIU4",
        # employment (production workers by sex × quarter)
        "PRDIH1T", "PRDIH2T", "PRDIH3T", "PRDIH4T",
        "PRDIM1T", "PRDIM2T", "PRDIM3T", "PRDIM4T",
        # skilled (experts)
        "ESPH1T", "ESPM1T",
        # totals
        "TOTHOM", "TOTMUJ",
        # wages
        "REMHPRDI", "REGPRDI", "APORPRDI",
        # taxes
        "CONIMP", "RENIMP", "TIMIMP", "PATIMP", "OTRIMP",
        # capital aggregates / outputs
        "VBP", "CI", "VA",
    ]
    print("\n=== Paper-critical fields presence ===")
    rows = []
    for f in critical:
        if f in inv.index:
            present = {y: inv.at[f, f"{y}_in"] for y in YEARS}
            parte = {}
            for y in YEARS:
                for p in [1, 2, 3]:
                    if inv.at[f, f"{y}_P{p}"]:
                        parte[y] = p
                        break
            rows.append({
                "field": f,
                "years_present": inv.at[f, "years_present"],
                "parte_by_year": ",".join(f"{y}:P{parte.get(y, '-')}" for y in YEARS),
            })
        else:
            rows.append({"field": f, "years_present": 0, "parte_by_year": "ABSENT"})
    crit = pd.DataFrame(rows)
    print(crit.to_string(index=False))
    crit.to_csv(OUT / "critical_field_presence.csv", index=False)


if __name__ == "__main__":
    main()
