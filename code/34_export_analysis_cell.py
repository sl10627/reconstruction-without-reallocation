"""Export the cell panel read by the R and Stata replications.

code/r/main_did.R and code/stata/main_did.do read one flat file instead of the
parquet panel. It holds the identifiers, treatment variables and logged outcomes of
data/enia_cell_2digit_quake.parquet (written by 15_earthquake_did.py), plus integer
ids for the industry x region (jr_id) and industry x year (jt_id) fixed effects,
numbered in order of first appearance.

It also writes data/python_benchmarks.csv: the Python estimates that the R and Stata
scripts must reproduce. Both scripts read this file and stop with an error if any of
their estimates differ, so the cross-language check does not rely on typed numbers.

Input:  data/enia_cell_2digit_quake.parquet, output/earthquake_did_main.csv,
        output/earthquake_event_study.csv, output/earthquake_event_study_pretrend.csv,
        output/wild_bootstrap.csv
Output: data/analysis_cell.csv, data/analysis_cell.dta, data/python_benchmarks.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"

COLS = ["ind2", "REGION", "ANIO", "year_c", "post", "mean_mmi", "high_mmi", "post_mmi",
        "post_high", "jr", "jt", "weight_sum", "n_obs", "log_n_est", "log_inv", "log_emp",
        "log_wage", "log_vbp"]


def main():
    q = pd.read_parquet(DATA / "enia_cell_2digit_quake.parquet")
    out = q[COLS].copy()
    out["jr_id"] = pd.factorize(out["jr"])[0] + 1
    out["jt_id"] = pd.factorize(out["jt"])[0] + 1
    assert not out.duplicated(["ind2", "REGION", "ANIO"]).any(), "one row per industry x region x year"

    out.to_csv(DATA / "analysis_cell.csv", index=False)
    # The .dta copy uses Stata's import-delimited conventions: lower-case names, long ints.
    dta = out.rename(columns=str.lower)
    for c in dta.select_dtypes("integer").columns:
        dta[c] = dta[c].astype("int32")
    dta.to_stata(DATA / "analysis_cell.dta", write_index=False, version=118)
    print(f"Wrote {DATA / 'analysis_cell.csv'} and .dta  shape={out.shape}")
    write_benchmarks()


def _one(df: pd.DataFrame, col: str, **eq) -> float:
    m = pd.Series(True, index=df.index)
    for k, v in eq.items():
        m &= df[k].astype(str).str.contains(v, regex=False) if isinstance(v, str) else df[k] == v
    assert m.sum() == 1, f"{eq} matched {m.sum()} rows"
    return float(df.loc[m, col].iloc[0])


def write_benchmarks():
    did = pd.read_csv(OUT / "earthquake_did_main.csv")
    es = pd.read_csv(OUT / "earthquake_event_study.csv")
    pt = pd.read_csv(OUT / "earthquake_event_study_pretrend.csv")
    wcb = pd.read_csv(OUT / "wild_bootstrap.csv")
    rows = [
        ("did_no_trend", _one(did, "beta", sample="MAIN", outcome="investment", treat="post_mmi", trends="none")),
        ("did_linear_trend", _one(did, "beta", sample="MAIN", outcome="investment", treat="post_mmi", trends="linear")),
        ("es_raw_2014", _one(es, "beta", outcome="investment", year=2014)),
        ("es_pretrend_2014", _one(pt, "beta", outcome="investment", year=2014)),
        ("es_pretrend_2015", _one(pt, "beta", outcome="investment", year=2015)),
        ("wcb_t_2014", _one(wcb, "t", estimate="2014 investment pulse")),
        ("wcb_p_2014", _one(wcb, "p_wcb", estimate="2014 investment pulse")),
    ]
    pd.DataFrame(rows, columns=["name", "value"]).to_csv(DATA / "python_benchmarks.csv", index=False)
    print(pd.DataFrame(rows, columns=["name", "value"]).to_string(index=False))


if __name__ == "__main__":
    main()
