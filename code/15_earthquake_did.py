"""2010 Maule earthquake (Feb 27, Mw 8.8) as an exogenous capital shock.

Within-industry across-region DiD (design option C):
    Y_{j,r,t} = alpha + beta * (Post_t x MMI_r) + gamma_{j,r} + delta_{j,t} + eps

  j = 2-digit ISIC industry, r = region (ENIA 1-15), t = year
  Post_t = 1[year >= 2010]
  MMI_r  = region area-weighted Modified Mercalli Intensity (USGS ShakeMap)
  gamma_{j,r} = industry x region FE  (absorbs MMI main effect + cell baseline)
  delta_{j,t} = industry x year FE    (absorbs Post + industry-wide annual shocks
                                       e.g. copper price for metals, wine demand for food)

beta interpretation: within the same industry-year, how much MORE did high-MMI
regions change relative to low-MMI regions.

Outcomes (Q2 = standard decomposition, 6 outcomes):
    1. log N_est        firm entry/exit margin
    2. log capital_stock  CBU_total (book value of capital in use)   [Channel 2]
    3. log investment    CBN_total (purchases of NEW capital goods)  [Channel 2]
    4. log EMPTOT        employment                                  [Channel 3]
    5. log wage          REGPRDI weighted mean (production wage)      [Channel 3]
    6. log VBP           gross output (aggregate)        [heavy caveat, see below]

Samples:
    MAIN        2-digit ISIC in {10 Food, 16 Wood, 24 Basic metals}  (clean controls)
    ROBUSTNESS  all manufacturing (2-digit 10-33)

--------------------------------------------------------------------------------
DATA-WINDOW CAVEATS (must travel with every result):
  * Pre-period = 2003,2004,2005,2007,2008,2009 (pre-2008 from the NUI-panel-era
    ENIA files, CIIU Rev.3). 2006 retrieved from INE; 2012 absent (low response).
    Six pre-quake years -> parallel-trends is now properly testable.
  * pre-2008 files key on NUI (stable panel id) and lack EMPTOT -> employment uses
    TOTHOM+TOTMUJ fallback; REGPRDI wage is absent pre-2008 (wage not extended).
  * 2008: occupational employment GRID broke (PRO* -> PRDI*) but EMPTOT total is
    intact; emp_eligible flag (>=2009) is conservative, we RELAX it for EMPTOT.
  * 2009: access-parser stored some numeric fields as byte blobs; decoded in
    step 02/04 to within ~10-30% (INE multiplicative noise + decode error).
  * VBP: vbp_eligible flag is >=2013 (comparability concern); VBP values DO exist
    2008+ so we run it, but the VBP result is the LEAST reliable -- flag in paper.
  * VA: literally zero before 2014 -> dropped from earthquake design entirely.
  * INVESTMENT field was RENAMED at the 2015 redesign (CBN* -> CB*), not dropped;
    unified here so investment now spans 2010-2017 (full post window). The 2015
    level break is national/common-within-industry-year -> absorbed by FE.
  * Capital STOCK: ENIA has no direct stock field (computed from purchases per the
    instructivo); VBU*/VR* are disposals/residual, not stock -> outcome dropped.
  * Clustering on region gives only ~15 clusters -> SEs are a lower bound on
    uncertainty; treat marginal significance with caution.
  * MMI missing for far north/south regions (1,2,11,12,15): genuinely unshaken,
    filled with floor MMI = 2.0 ("felt/weak") so they serve as low-MMI controls.

Outputs:
  data/enia_cell_2digit_quake.parquet   the cell panel used here
  output/earthquake_did_main.csv         main spec coefficients (both samples)
  output/earthquake_event_study.csv      event-study coefficients
  output/earthquake_placebo.csv          fake-2014 placebo
  output/figs/E3_event_study.png         event-study plot (all outcomes)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# INVESTMENT (gross purchases of capital goods). The field was RENAMED in the
# 2015 form redesign, NOT discontinued:
#   2008-2014: CBN*  (compras de bienes nuevos)
#   2015-2017: CB*   (compras de bienes, new schema)
# We unify them into one series spanning 2008-2017. The national definitional
# rescaling at 2015 is common across regions within each industry-year, so it is
# fully absorbed by the industry x year FE -> the DiD beta is unaffected.
CBN_FIELDS_OLD = ["CBNTER", "CBNEDI", "CBNMAQ", "CBNVEH", "CBNMUE", "CBNSOF", "CBNOAT", "CBNOTRI"]
CB_FIELDS_NEW = ["CBTER", "CBEDI", "CBMAQ", "CBVEH", "CBSOF", "CBOAT", "CBOTRI"]
# NOTE: ENIA has NO direct capital-stock field. The instructivo states capital is
# "no se mide directamente ... se calcula a partir de las compras de bienes".
# The VBU*/VR* fields are tiny (smaller than annual depreciation) -> disposals/
# residual value, NOT a stock. So "capital stock" is dropped as an outcome; the
# capital-rebuilding margin is captured by the INVESTMENT flow (CBN/CB) above.

MAIN_IND2 = ["10", "16", "24"]   # Food, Wood, Basic metals
MMI_FLOOR = 2.0                  # fill for unshaken far regions

# decomposition order; (column, label, is_mean) -- is_mean=True -> weighted mean, else weighted sum
OUTCOMES = [
    ("n_est",  "log N_est (entry/exit)",      None),   # built from weights directly
    ("inv",    "log investment (CBN/CB)",     False),  # capital-rebuilding margin
    ("emp",    "log employment (EMPTOT)",     False),
    ("wage",   "log wage (REGPRDI)",          True),
    ("vbp",    "log gross output (VBP)*",     False),
]


def build_cell_panel() -> pd.DataFrame:
    pool = pd.read_parquet(DATA / "enia_pooled.parquet")
    print(f"Pooled loaded: {pool.shape}")

    pool = pool[pool["ind_rev4"].notna() & pool["REGION"].notna() & pool["ANIO"].notna()].copy()
    pool["ind2"] = pool["ind_rev4"].astype(str).str[:2]
    pool["REGION"] = pd.to_numeric(pool["REGION"], errors="coerce")
    pool["ANIO"] = pd.to_numeric(pool["ANIO"], errors="coerce").astype(int)
    pool["w"] = pd.to_numeric(pool["ind_weight"], errors="coerce").fillna(0.0)

    # build composite level variables on the establishment rows.
    # INVESTMENT: CBN* for <=2014, CB* for >=2015 (renamed field, unified series)
    inv_old = pool[[c for c in CBN_FIELDS_OLD if c in pool.columns]].apply(
        pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    inv_new = pool[[c for c in CB_FIELDS_NEW if c in pool.columns]].apply(
        pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    pool["inv_lvl"] = np.where(pool["ANIO"] <= 2014, inv_old, inv_new)
    # EMPTOT total; pre-2008 NUI files lack EMPTOT -> fall back to TOTHOM+TOTMUJ
    emp = pd.to_numeric(pool["EMPTOT"], errors="coerce") if "EMPTOT" in pool.columns else pd.Series(np.nan, index=pool.index)
    emp_hm = (pd.to_numeric(pool.get("TOTHOM"), errors="coerce").fillna(0)
              + pd.to_numeric(pool.get("TOTMUJ"), errors="coerce").fillna(0))
    pool["emp_lvl"] = emp.where(emp > 0, emp_hm.where(emp_hm > 0))
    pool["vbp_lvl"] = pd.to_numeric(pool["VBP"], errors="coerce")
    pool["wage_lvl"] = pd.to_numeric(pool["REGPRDI"], errors="coerce")

    def wsum(v: pd.Series, w: pd.Series) -> float:
        m = v.notna() & (w > 0)
        return (v[m] * w[m]).sum() if m.any() else np.nan

    def wmean(v: pd.Series, w: pd.Series) -> float:
        m = v.notna() & (v > 0) & (w > 0)
        return (v[m] * w[m]).sum() / w[m].sum() if m.any() else np.nan

    rows = []
    for (i2, reg, yr), g in pool.groupby(["ind2", "REGION", "ANIO"]):
        w = g["w"]
        rows.append({
            "ind2": i2, "REGION": int(reg), "ANIO": int(yr),
            "n_obs": len(g),
            "weight_sum": w.sum(),
            "n_est": w.sum(),                      # effective establishment count
            "inv":  wsum(g["inv_lvl"], w),
            "emp":  wsum(g["emp_lvl"], w),
            "vbp":  wsum(g["vbp_lvl"], w),
            "wage": wmean(g["wage_lvl"], w),
        })
    cell = pd.DataFrame(rows)
    print(f"Cell panel (2-digit x region x year): {cell.shape}")

    # merge MMI
    shake = pd.read_csv(DATA / "shakemap_by_region.csv")
    shake = shake.rename(columns={"enia_region": "REGION"})[["REGION", "mean_mmi", "frac_area_mmi_ge_7"]]
    cell = cell.merge(shake, on="REGION", how="left")
    n_missing = cell["mean_mmi"].isna().sum()
    cell["mean_mmi"] = cell["mean_mmi"].fillna(MMI_FLOOR)
    cell["frac_area_mmi_ge_7"] = cell["frac_area_mmi_ge_7"].fillna(0.0)
    print(f"  MMI merged; filled {n_missing} cell-rows in unshaken regions with floor {MMI_FLOOR}")

    # treatment vars and FE keys
    cell["post"] = (cell["ANIO"] >= 2010).astype(int)
    cell["year_c"] = cell["ANIO"] - 2009          # centered at reference year for region trends
    cell["high_mmi"] = (cell["mean_mmi"] >= 6.0).astype(int)
    cell["post_mmi"] = cell["post"] * cell["mean_mmi"]
    cell["post_high"] = cell["post"] * cell["high_mmi"]
    cell["jr"] = cell["ind2"] + "_r" + cell["REGION"].astype(str)
    cell["jt"] = cell["ind2"] + "_t" + cell["ANIO"].astype(str)

    # log outcomes (drop nonpositive)
    for col in ["n_est", "inv", "emp", "vbp", "wage"]:
        cell[f"log_{col}"] = np.where(cell[col] > 0, np.log(cell[col]), np.nan)

    cell.to_parquet(DATA / "enia_cell_2digit_quake.parquet", index=False)
    print(f"Wrote {DATA / 'enia_cell_2digit_quake.parquet'}")
    return cell


def _fe_ok(sub: pd.DataFrame, ycol: str) -> pd.DataFrame:
    """Keep only cells with valid outcome, and FE groups with >=2 obs so the
    industry x region and industry x year FE are estimable / not singleton."""
    sub = sub.dropna(subset=[ycol]).copy()
    for key in ["jr", "jt"]:
        counts = sub[key].value_counts()
        sub = sub[sub[key].isin(counts[counts >= 2].index)]
    return sub


def run_main(cell: pd.DataFrame, treat: str, trend: str = "none") -> pd.DataFrame:
    """treat in {'post_mmi','post_high'}. trend in {'none','linear','quad'} adds
    region-specific linear / quadratic time trends (full-sample) to absorb the
    differential secular growth of the (urban, central) high-MMI regions.
    Quadratic is a conventional nonlinear-trend robustness; HonestDiD (script 24)
    is the principled sensitivity analysis. Returns coefficient table over samples."""
    trend_term = {"none": "", "linear": " + C(REGION):year_c",
                  "quad": " + C(REGION):year_c + C(REGION):year_c2"}[trend]
    results = []
    samples = {
        "MAIN (Food/Wood/Metals)": cell[cell["ind2"].isin(MAIN_IND2)],
        "ALL manufacturing": cell,
    }
    for sample_name, sdf in samples.items():
        for col, label, _ in OUTCOMES:
            ycol = f"log_{col}"
            sub = _fe_ok(sdf, ycol).copy()
            sub["year_c2"] = sub["year_c"] ** 2
            n_reg = sub["REGION"].nunique()
            row0 = {"sample": sample_name, "outcome": label, "treat": treat,
                    "trends": trend, "n_regions": n_reg}
            if len(sub) < 25 or n_reg < 3:
                results.append({**row0, "beta": np.nan, "se": np.nan, "p": np.nan,
                                "n": len(sub), "note": "insufficient cells"})
                continue
            try:
                m = smf.wls(f"{ycol} ~ {treat} + C(jr) + C(jt){trend_term}", data=sub,
                            weights=sub["weight_sum"].clip(lower=1)).fit(
                    cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
                results.append({**row0, "beta": m.params.get(treat, np.nan),
                                "se": m.bse.get(treat, np.nan), "p": m.pvalues.get(treat, np.nan),
                                "n": int(m.nobs), "note": ""})
            except Exception as e:  # noqa: BLE001
                results.append({**row0, "beta": np.nan, "se": np.nan, "p": np.nan,
                                "n": len(sub), "note": f"error: {type(e).__name__}"})
    return pd.DataFrame(results)


def run_event_study(cell: pd.DataFrame, trends: bool = False) -> pd.DataFrame:
    """year x MMI interactions, ref year = 2009. MAIN sample. trends=True adds
    region linear trends so each year's coef is a deviation from the region trend."""
    years = [2003, 2004, 2005, 2006, 2007, 2008, 2010, 2011, 2013, 2014, 2015, 2016, 2017]  # 2009 = ref
    trend_term = " + C(REGION):year_c" if trends else ""
    sdf = cell[cell["ind2"].isin(MAIN_IND2)].copy()
    for y in years:
        sdf[f"mmi_{y}"] = sdf["mean_mmi"] * (sdf["ANIO"] == y).astype(int)
    terms = " + ".join(f"mmi_{y}" for y in years)

    rows = []
    for col, label, _ in OUTCOMES:
        ycol = f"log_{col}"
        sub = _fe_ok(sdf, ycol)
        if len(sub) < 25 or sub["REGION"].nunique() < 3:
            continue
        try:
            m = smf.wls(f"{ycol} ~ {terms} + C(jr) + C(jt){trend_term}", data=sub,
                        weights=sub["weight_sum"].clip(lower=1)).fit(
                cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
            for y in years:
                t = f"mmi_{y}"
                rows.append({"outcome": label, "year": y,
                             "beta": m.params.get(t, np.nan), "se": m.bse.get(t, np.nan),
                             "p": m.pvalues.get(t, np.nan)})
            rows.append({"outcome": label, "year": 2009, "beta": 0.0, "se": 0.0, "p": np.nan})
        except Exception as e:  # noqa: BLE001
            print(f"  event-study {label}: {type(e).__name__}")
    return pd.DataFrame(rows).sort_values(["outcome", "year"])


def run_placebo(cell: pd.DataFrame) -> pd.DataFrame:
    """Falsification: drop the real shock, keep post-quake years only (2010-2017),
    pretend the 'shock' happened in 2014. beta should be ~0."""
    sdf = cell[(cell["ind2"].isin(MAIN_IND2)) & (cell["ANIO"] >= 2010)].copy()
    sdf["fake_post"] = (sdf["ANIO"] >= 2014).astype(int)
    sdf["fake_post_mmi"] = sdf["fake_post"] * sdf["mean_mmi"]
    rows = []
    for col, label, _ in OUTCOMES:
        ycol = f"log_{col}"
        sub = _fe_ok(sdf, ycol)
        if len(sub) < 25 or sub["REGION"].nunique() < 3:
            continue
        try:
            m = smf.wls(f"{ycol} ~ fake_post_mmi + C(jr) + C(jt)", data=sub,
                        weights=sub["weight_sum"].clip(lower=1)).fit(
                cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
            rows.append({"outcome": label, "beta": m.params.get("fake_post_mmi", np.nan),
                         "se": m.bse.get("fake_post_mmi", np.nan),
                         "p": m.pvalues.get("fake_post_mmi", np.nan), "n": int(m.nobs)})
        except Exception as e:  # noqa: BLE001
            print(f"  placebo {label}: {type(e).__name__}")
    return pd.DataFrame(rows)


PRE_YEARS = [2003, 2004, 2005, 2006, 2007, 2008, 2009]
ES_YEARS = [2003, 2004, 2005, 2006, 2007, 2008, 2010, 2011, 2013, 2014, 2015, 2016, 2017]  # 2009 = ref


def run_event_study_pretrend(cell: pd.DataFrame) -> pd.DataFrame:
    """Event study where region linear trends are estimated on the PRE-period only
    (2003-2009) and projected forward, so the post reconstruction spike does NOT
    contaminate the trend slope (avoids the attenuation of the full-sample-trend
    version). Two-step: (1) remove industry x region pre-mean level; (2) per-region
    pre-period slope on year_c; (3) detrend all years and run the MMI x year event
    study on residuals. NOTE: step-3 SEs treat the detrended series as data and do
    not propagate step-1/2 estimation error -> SEs are a (mild) lower bound."""
    w = "weight_sum"
    sdf = cell[cell["ind2"].isin(MAIN_IND2)].copy()
    rows = []
    for col, label, _ in OUTCOMES:
        ycol = f"log_{col}"
        sub = _fe_ok(sdf, ycol).copy()
        pre = sub[sub["ANIO"].isin(PRE_YEARS)]
        if pre.empty:
            continue
        # (1) industry x region level from pre-period (weighted mean)
        jr_mean = (pre.groupby("jr")
                      .apply(lambda d: (d[ycol] * d[w]).sum() / d[w].sum(), include_groups=False)
                      .to_dict())
        sub = sub[sub["jr"].isin(jr_mean)].copy()
        sub["_dm"] = sub[ycol] - sub["jr"].map(jr_mean)
        # (2) per-region linear slope on pre-period
        slopes = {}
        for r, d in sub[sub["ANIO"].isin(PRE_YEARS)].groupby("REGION"):
            if d["year_c"].nunique() >= 2 and len(d) >= 3:
                mm = sm.WLS(d["_dm"], sm.add_constant(d["year_c"]),
                            weights=d[w].clip(lower=1)).fit()
                slopes[r] = mm.params.get("year_c", 0.0)
            else:
                slopes[r] = 0.0
        # (3) detrend all years, event study on residual
        sub["_detr"] = sub["_dm"] - sub["REGION"].map(slopes) * sub["year_c"]
        for y in ES_YEARS:
            sub[f"mmi_{y}"] = sub["mean_mmi"] * (sub["ANIO"] == y).astype(int)
        terms = " + ".join(f"mmi_{y}" for y in ES_YEARS)
        if sub["REGION"].nunique() < 3:
            continue
        try:
            m = smf.wls(f"_detr ~ {terms} + C(jt)", data=sub,
                        weights=sub[w].clip(lower=1)).fit(
                cov_type="cluster", cov_kwds={"groups": sub["REGION"]})
            for y in ES_YEARS:
                t = f"mmi_{y}"
                rows.append({"outcome": label, "year": y, "beta": m.params.get(t, np.nan),
                             "se": m.bse.get(t, np.nan), "p": m.pvalues.get(t, np.nan)})
            rows.append({"outcome": label, "year": 2009, "beta": 0.0, "se": 0.0, "p": np.nan})
        except Exception as e:  # noqa: BLE001
            print(f"  pretrend ES {label}: {type(e).__name__}")
    return pd.DataFrame(rows).sort_values(["outcome", "year"])


def plot_event_study(es: pd.DataFrame, fname: str = "E3_event_study.png", suffix: str = ""):
    # drop degenerate points (no data: beta==se==0 and not the reference year)
    es = es[(es["year"] == 2009) | ~((es["se"] == 0) & (es["beta"].abs() < 1e-9))].copy()
    labels = [l for _, l, _ in OUTCOMES if l in es["outcome"].unique()]
    n = len(labels)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.5 * nrow), squeeze=False)
    for ax, label in zip(axes.flat, labels):
        d = es[es["outcome"] == label].sort_values("year")
        ax.errorbar(d["year"], d["beta"], yerr=1.96 * d["se"], marker="o",
                    capsize=3, color="darkred", linewidth=1.2)
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(2009.5, color="navy", ls="--", lw=1, label="earthquake (2010)")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("year"); ax.set_ylabel("beta (year x MMI)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle("Earthquake event study: differential outcome by region MMI, ref = 2009\n"
                 "MAIN (Food/Wood/Metals); industry x region + industry x year FE; "
                 f"pre 2003-2009 (NUI era){suffix}", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=140)
    plt.close(fig)
    print(f"Wrote {FIGS / fname}")


def main():
    cell = build_cell_panel()

    print("\n=== MAIN spec: continuous Post x MMI  (none / linear / quadratic trends) ===")
    main_c = pd.concat([run_main(cell, "post_mmi", trend=t)
                        for t in ("none", "linear", "quad")], ignore_index=True)
    print(main_c[["sample", "outcome", "trends", "beta", "se", "p", "n"]].round(4).to_string(index=False))

    print("\n=== MAIN spec: binary Post x high-MMI(>=6)  (none / linear / quadratic trends) ===")
    main_b = pd.concat([run_main(cell, "post_high", trend=t)
                        for t in ("none", "linear", "quad")], ignore_index=True)
    print(main_b[["sample", "outcome", "trends", "beta", "se", "p", "n"]].round(4).to_string(index=False))

    pd.concat([main_c, main_b], ignore_index=True).to_csv(OUT / "earthquake_did_main.csv", index=False)
    print(f"\nWrote {OUT / 'earthquake_did_main.csv'}")

    print("\n=== Event study (continuous MMI, ref 2009) — RAW (no trend) ===")
    es = run_event_study(cell, trends=False)
    es.to_csv(OUT / "earthquake_event_study.csv", index=False)
    print(es.round(4).to_string(index=False))
    plot_event_study(es, "E3_event_study.png", "  [no detrend]")

    print("\n=== Event study (continuous MMI, ref 2009) — DETRENDED (full-sample region trends) ===")
    es_tr = run_event_study(cell, trends=True)
    es_tr.to_csv(OUT / "earthquake_event_study_detrended.csv", index=False)
    print(es_tr.round(4).to_string(index=False))
    plot_event_study(es_tr, "E4_event_study_detrended.png", "  + region trends, full-sample (conservative)")

    print("\n=== Event study (continuous MMI, ref 2009) — PRE-TREND PROJECTED (trend from 2003-09 only) ===")
    es_pt = run_event_study_pretrend(cell)
    es_pt.to_csv(OUT / "earthquake_event_study_pretrend.csv", index=False)
    print(es_pt.round(4).to_string(index=False))
    plot_event_study(es_pt, "E5_event_study_pretrend.png", "  + region trends from PRE-period only (projected)")

    # --- bracket the key 2014 investment coefficient across the three trend treatments ---
    def coef2014(df):
        r = df[(df["outcome"].str.startswith("log investment")) & (df["year"] == 2014)]
        return (float(r["beta"].iloc[0]), float(r["se"].iloc[0]), float(r["p"].iloc[0])) if len(r) else (np.nan,) * 3
    print("\n=== 2014 investment x MMI coefficient under three trend treatments ===")
    for name, df in [("raw (no trend, UPPER)", es), ("full-sample trend (LOWER)", es_tr),
                     ("pre-period trend (PREFERRED)", es_pt)]:
        b, s, p = coef2014(df)
        print(f"  {name:32s} beta={b:+.3f}  se={s:.3f}  p={p:.3f}")

    print("\n=== Placebo: fake 2014 shock (post-quake sample only) ===")
    plc = run_placebo(cell)
    plc.to_csv(OUT / "earthquake_placebo.csv", index=False)
    print(plc.round(4).to_string(index=False))

    print("\nDONE. Caveats: see module docstring (pre 2003-09 NUI era; VBP weakest; "
          "~15 region clusters; wage not extended pre-2008).")


if __name__ == "__main__":
    main()
