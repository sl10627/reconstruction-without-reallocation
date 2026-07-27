"""Extract all FUSION_PARTE tables from the 11 ENIA ACCDB files into CSV.

Output: raw/{year}_PARTE{N}.csv  (one file per part per year, 33 total)
"""

import re
import sys
from pathlib import Path

import pandas as pd
from access_parser import AccessParser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
RAW_DIR.mkdir(exist_ok=True)

YEAR_RE = re.compile(r"20\d{2}")
PARTE_RE = re.compile(r"^FUSION_PARTE(\d+)(?:_\d{4})?$", re.IGNORECASE)


def year_from_filename(name: str) -> str:
    m = YEAR_RE.search(name)
    if not m:
        raise ValueError(f"no year in filename: {name}")
    return m.group(0)


def extract_one(accdb_path: Path) -> list[tuple[str, int, int]]:
    """Returns list of (out_filename, n_rows, n_cols) tuples."""
    year = year_from_filename(accdb_path.name)
    db = AccessParser(str(accdb_path))
    results = []
    for table_name in db.catalog:
        m = PARTE_RE.match(table_name)
        if not m:
            continue
        part_num = int(m.group(1))
        parsed = db.parse_table(table_name)
        df = pd.DataFrame(parsed)
        out = RAW_DIR / f"{year}_PARTE{part_num}.csv"
        df.to_csv(out, index=False)
        results.append((out.name, len(df), df.shape[1]))
    return results


def main() -> int:
    accdbs = sorted(PROJECT_ROOT.glob("*.accdb"))
    if not accdbs:
        print("no .accdb files found in", PROJECT_ROOT)
        return 1
    print(f"found {len(accdbs)} ACCDB files\n")
    summary = []
    for f in accdbs:
        print(f"-- {f.name}")
        try:
            for out_name, nr, nc in extract_one(f):
                print(f"   {out_name:30s}  rows={nr:>6d}  cols={nc:>4d}")
                summary.append((out_name, nr, nc))
        except Exception as e:
            print(f"   ERROR: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"\ntotal CSVs written: {len(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
