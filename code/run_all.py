"""Run the full replication pipeline, from the raw ENIA databases to the paper's
tables, figures and prose numbers.

    python code/run_all.py                    # everything, in order
    python code/run_all.py --from 15          # resume from a step (by script number)
    python code/run_all.py --only 22 17       # run only these steps
    python code/run_all.py --exploratory      # also run the exploratory scripts

The script numbers identify each script, and PIPELINE below fixes the run order, which
follows the data dependencies (for example, 32 builds the CASEN panel that 25, 27
and 28 read). Each step's console output is saved to output/logs/<script>.log. The
last step, verify_numbers.py, fails the run if any number quoted in the paper text
differs from the freshly generated outputs.

Inputs that are not produced by any script (see README.md, "Data"):
  *-base-enia.accdb                 ENIA annual databases, at the repository root
  crosswalks/ISIC31_ISIC4.txt       UN ISIC Rev. 3.1 -> Rev. 4 concordance
  geodata/gadm_chile/, geodata/shakemap_shape/   GADM regions, USGS ShakeMap polygons
  data/casen_{2006,...,2017}.dta    CASEN household surveys
  data/_casen_extract/casen2003.dta CASEN 2003, extracted from data/casen2003stata.rar
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
LOGS = ROOT / "output" / "logs"

# (script, what it produces). Order matters.
PIPELINE = [
    ("01_extract_accdb.py",            "raw/<year>_PARTE<k>.csv from the .accdb files"),
    ("04_harmonize.py",                "data/wide_<year>.parquet"),
    ("05_build_crosswalk_weights.py",  "crosswalks/rev31_to_rev4_share_weights.csv"),
    ("06_build_pooled.py",             "data/enia_pooled.parquet"),
    ("13_earthquake_intensity.py",     "data/shakemap_by_region.csv"),
    ("31_remake_fig1.py",              "Figure 1 (output/figs/E2_chile_mmi_map_zoom.png)"),
    ("15_earthquake_did.py",           "cell panel, main DiD, event studies (Figure 2)"),
    ("16_robustness.py",               "robustness of the 2014 pulse"),
    ("18_labor_outcomes.py",           "labor outcomes, data/enia_cell_labor.parquet"),
    ("19_asset_decomposition.py",      "asset decomposition (Figure 3)"),
    ("20_net_investment.py",           "net investment test"),
    ("21_wage_decomposition.py",       "wage decomposition (Figure 4)"),
    ("33_wage_mechanism_checks.py",    "wage-mechanism checks table"),
    ("22_wild_bootstrap.py",           "wild cluster bootstrap"),
    ("24_honestdid.py",                "HonestDiD sensitivity (Figure 6)"),
    ("32_build_casen_panel.py",        "data/casen_panel.parquet"),
    ("26_prep_casen2003.py",           "data/casen2003_slim.parquet"),
    ("25_casen_migration.py",          "CASEN migration check (Figure 5)"),
    ("27_casen_construction.py",       "CASEN construction-sector check"),
    ("28_agri_placebo.py",             "agricultural placebo"),
    ("29_incidence.py",                "incidence decomposition"),
    ("30_spillover.py",                "spatial spillover test"),
    ("34_export_analysis_cell.py",     "data/analysis_cell.csv/.dta for R and Stata"),
    ("17_make_tables.py",              "paper/tables/*.tex"),
    ("verify_numbers.py",              "checks every number in the paper text"),
]

# Earlier project directions and diagnostics. Nothing in the paper depends on them.
EXPLORATORY = [
    "02_decode_2018.py", "03_schema_compare.py", "07_descriptive_plots.py",
    "08_long_difference.py", "09_investment_channel.py", "10_robustness.py",
    "11_entry_exit_explore.py", "12_manufacturing_aggregate.py",
    "14_industry_by_region.py", "23_hours_margin.py",
]


def step_id(script: str) -> str:
    return script.split("_", 1)[0]


def run(script: str) -> float:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / script.replace(".py", ".log")
    t0 = time.time()
    with log.open("w") as fh:
        proc = subprocess.run([sys.executable, str(CODE / script)], cwd=ROOT,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        fh.write(proc.stdout)
    secs = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        sys.exit(f"FAILED: {script} (exit {proc.returncode}), full log in {log}")
    return secs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", help="start at this step number, e.g. 15")
    ap.add_argument("--only", nargs="+", help="run only these step numbers")
    ap.add_argument("--exploratory", action="store_true", help="also run the exploratory scripts")
    args = ap.parse_args()

    steps = [s for s, _ in PIPELINE]
    if args.start:
        ids = [step_id(s) for s in steps]
        if args.start not in ids:
            sys.exit(f"unknown step {args.start}. Steps are {', '.join(ids)}")
        steps = steps[ids.index(args.start):]
    if args.only:
        steps = [s for s in steps if step_id(s) in args.only]
    if args.exploratory:
        steps += EXPLORATORY

    # Folders the scripts write into. Most scripts assume they already exist.
    for d in ["raw", "data", "output/figs", "output/logs", "paper/tables"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    total = time.time()
    for s in steps:
        print(f"-> {s:34s}", end="", flush=True)
        print(f"{run(s):7.1f}s")
    print(f"Done: {len(steps)} steps in {(time.time() - total) / 60:.1f} min. Logs in {LOGS}")


if __name__ == "__main__":
    main()
