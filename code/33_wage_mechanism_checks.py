"""Wage-mechanism checks: alternative explanations for the unskilled-wage response.

C1  composition: 2014 event-study coefficients for establishments, employment and
    gross output, read from 15_earthquake_did.py's pretrend event study.
C2  unskilled-wage baseline, plus a fake 2014 shock inside the post period.
C3  the unskilled-wage effect with leave-one-industry-out region-year manufacturing
    controls.
C4  skilled wage (internal placebo) and the skill premium.
C5  food only vs the reconstruction-adjacent industries (wood, basic metals).

This script was previously kept outside the repository. It now runs as part of
run_all.py after 15_earthquake_did.py.

Input:  data/enia_pooled.parquet, data/shakemap_by_region.csv,
        data/enia_cell_2digit_quake.parquet, output/earthquake_event_study_pretrend.csv
Output: output/wage_mechanism_checks.csv, output/wage_mechanism_event_studies.csv,
        paper/tables/tab_wage_mechanism_checks.tex,
        output/figs/E10_wage_mechanism_checks.png
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
PAPER_TABLES = ROOT / "paper" / "tables"

MAIN_IND2 = ["10", "16", "24"]
RECON_ADJ = ["16", "24"]
PRE_YEARS = [2003, 2004, 2005, 2007, 2008, 2009]
ES_YEARS = [2003, 2004, 2005, 2007, 2008, 2010, 2011, 2013, 2014, 2015, 2016]
W = "weight_sum"
MMI_FLOOR = 2.0


def pstars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt(beta: float, p: float) -> str:
    if pd.isna(beta):
        return "---"
    return f"{beta:.3f}{pstars(p)}"


def fmt_n(n: float) -> str:
    if pd.isna(n):
        return "---"
    return str(int(n))


def build_wage_cell() -> pd.DataFrame:
    p = pd.read_parquet(DATA / "enia_pooled.parquet")
    p = p[p["ind_rev4"].notna() & p["REGION"].notna() & p["ANIO"].notna()].copy()
    p["ind2"] = p["ind_rev4"].astype(str).str[:2]
    p["REGION"] = pd.to_numeric(p["REGION"], errors="coerce").astype(int)
    p["ANIO"] = pd.to_numeric(p["ANIO"], errors="coerce").astype(int)
    p["w"] = pd.to_numeric(p["ind_weight"], errors="coerce").fillna(0.0)
    p["wskill"] = pd.to_numeric(p.get("REGTESP"), errors="coerce")
    p["wunsk"] = pd.to_numeric(p.get("REGNOCALD"), errors="coerce")

    def wmean_pos(v: pd.Series, w: pd.Series) -> float:
        m = v.notna() & (v > 0) & (w > 0)
        return (v[m] * w[m]).sum() / w[m].sum() if m.any() else np.nan

    rows = []
    for (i2, reg, yr), g in p.groupby(["ind2", "REGION", "ANIO"]):
        ws = wmean_pos(g["wskill"], g["w"])
        wu = wmean_pos(g["wunsk"], g["w"])
        rows.append(
            {
                "ind2": i2,
                "REGION": int(reg),
                "ANIO": int(yr),
                "weight_sum": g["w"].sum(),
                "log_wskill": np.log(ws) if pd.notna(ws) and ws > 0 else np.nan,
                "log_wunsk": np.log(wu) if pd.notna(wu) and wu > 0 else np.nan,
                "log_skillprem": np.log(ws / wu)
                if pd.notna(ws) and pd.notna(wu) and ws > 0 and wu > 0
                else np.nan,
            }
        )
    cell = pd.DataFrame(rows)
    sh = pd.read_csv(DATA / "shakemap_by_region.csv").rename(columns={"enia_region": "REGION"})
    cell = cell.merge(sh[["REGION", "mean_mmi"]], on="REGION", how="left", validate="m:1")
    cell["mean_mmi"] = cell["mean_mmi"].fillna(MMI_FLOOR)
    cell["year_c"] = cell["ANIO"] - 2009
    cell["high_mmi"] = (cell["mean_mmi"] >= 6).astype(int)
    cell["jr"] = cell["ind2"] + "_r" + cell["REGION"].astype(str)
    cell["jt"] = cell["ind2"] + "_t" + cell["ANIO"].astype(str)
    return cell


def fe_ok(sub: pd.DataFrame, ycol: str) -> pd.DataFrame:
    sub = sub.dropna(subset=[ycol]).copy()
    for key in ["jr", "jt"]:
        vc = sub[key].value_counts()
        sub = sub[sub[key].isin(vc[vc >= 2].index)]
    return sub


def detrend(cell: pd.DataFrame, ycol: str, inds: list[str] | None = None) -> pd.DataFrame:
    sub = cell.copy()
    if inds is not None:
        sub = sub[sub["ind2"].isin(inds)]
    sub = fe_ok(sub, ycol)
    pre = sub[sub["ANIO"].isin(PRE_YEARS)].copy()
    if pre.empty:
        return pd.DataFrame()
    jr_mean = (
        pre.groupby("jr")
        .apply(lambda d: (d[ycol] * d[W]).sum() / d[W].sum(), include_groups=False)
        .to_dict()
    )
    sub = sub[sub["jr"].isin(jr_mean)].copy()
    sub["_dm"] = sub[ycol] - sub["jr"].map(jr_mean)
    slopes = {}
    for r, d in sub[sub["ANIO"].isin(PRE_YEARS)].groupby("REGION"):
        if d["year_c"].nunique() >= 2 and len(d) >= 3:
            m = sm.WLS(d["_dm"], sm.add_constant(d["year_c"]), weights=d[W].clip(lower=1)).fit()
            slopes[r] = m.params.get("year_c", 0.0)
        else:
            slopes[r] = 0.0
    sub["_detr"] = sub["_dm"] - sub["REGION"].map(slopes).fillna(0.0) * sub["year_c"]
    return sub


def estimate_event(sub: pd.DataFrame, label: str, controls: list[str] | None = None) -> pd.DataFrame:
    controls = controls or []
    sub = sub.copy()
    years = [y for y in ES_YEARS if (sub["ANIO"] == y).any()]
    for y in years:
        sub[f"m_{y}"] = sub["mean_mmi"] * (sub["ANIO"] == y).astype(int)
    rhs = " + ".join([f"m_{y}" for y in years] + controls + ["C(jt)"])
    m = smf.wls(f"_detr ~ {rhs}", data=sub, weights=sub[W].clip(lower=1)).fit(
        cov_type="cluster", cov_kwds={"groups": sub["REGION"]}
    )
    rows = []
    for y in years:
        rows.append(
            {
                "check": label,
                "year": y,
                "beta": m.params.get(f"m_{y}", np.nan),
                "se": m.bse.get(f"m_{y}", np.nan),
                "p": m.pvalues.get(f"m_{y}", np.nan),
                "n": int(m.nobs),
            }
        )
    rows.append({"check": label, "year": 2009, "beta": 0.0, "se": 0.0, "p": np.nan, "n": int(m.nobs)})
    return pd.DataFrame(rows)


def estimate_avg_and_2014(sub: pd.DataFrame, label: str, controls: list[str] | None = None) -> dict:
    controls = controls or []
    sub = sub.copy()
    sub["post_mmi"] = (sub["ANIO"] >= 2010).astype(int) * sub["mean_mmi"]
    sub["m2014"] = (sub["ANIO"] == 2014).astype(int) * sub["mean_mmi"]
    post_rhs = " + ".join(["post_mmi"] + controls + ["C(jt)"])
    y14_rhs = " + ".join(["m2014"] + controls + ["C(jt)"])
    a = smf.wls(f"_detr ~ {post_rhs}", data=sub, weights=sub[W].clip(lower=1)).fit(
        cov_type="cluster", cov_kwds={"groups": sub["REGION"]}
    )
    b = smf.wls(f"_detr ~ {y14_rhs}", data=sub, weights=sub[W].clip(lower=1)).fit(
        cov_type="cluster", cov_kwds={"groups": sub["REGION"]}
    )
    return {
        "check": label,
        "avg_beta": a.params.get("post_mmi", np.nan),
        "avg_se": a.bse.get("post_mmi", np.nan),
        "avg_p": a.pvalues.get("post_mmi", np.nan),
        "y2014_beta": b.params.get("m2014", np.nan),
        "y2014_se": b.bse.get("m2014", np.nan),
        "y2014_p": b.pvalues.get("m2014", np.nan),
        "n": int(a.nobs),
    }


def add_region_year_controls(wage_cell: pd.DataFrame) -> pd.DataFrame:
    q = pd.read_parquet(DATA / "enia_cell_2digit_quake.parquet")
    cols = ["ind2", "REGION", "ANIO", "n_est", "emp", "vbp", "inv", "wage"]
    q = q[cols].copy()
    for c in ["n_est", "emp", "vbp", "inv", "wage"]:
        q[c] = pd.to_numeric(q[c], errors="coerce")
    total = q.groupby(["REGION", "ANIO"])[["n_est", "emp", "vbp", "inv"]].sum(min_count=1).reset_index()
    w_num = q.assign(wage_num=q["wage"] * q["n_est"]).groupby(["REGION", "ANIO"])[["wage_num", "n_est"]].sum().reset_index()
    total = total.merge(w_num, on=["REGION", "ANIO"], suffixes=("", "_w"), validate="1:1")
    total["wage"] = total["wage_num"] / total["n_est_w"]
    own = q.rename(columns={c: f"own_{c}" for c in ["n_est", "emp", "vbp", "inv", "wage"]})
    out = wage_cell.merge(total[["REGION", "ANIO", "n_est", "emp", "vbp", "inv", "wage"]], on=["REGION", "ANIO"], how="left", validate="m:1")
    out = out.merge(own, on=["ind2", "REGION", "ANIO"], how="left", validate="m:1")
    for c in ["n_est", "emp", "vbp", "inv"]:
        loo = out[c] - out[f"own_{c}"].fillna(0)
        out[f"loo_log_{c}"] = np.where(loo > 0, np.log(loo), np.nan)
    # Leave-one-industry-out wage is noisy, so keep it as a broad regional manufacturing wage proxy.
    out["ry_log_wage"] = np.where(out["wage"] > 0, np.log(out["wage"]), np.nan)
    for c in ["loo_log_n_est", "loo_log_emp", "loo_log_vbp", "loo_log_inv", "ry_log_wage"]:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
        out[c] = out[c].fillna(out[c].median())
    return out


def placebo_post_2014(sub: pd.DataFrame, label: str) -> dict:
    d = sub[sub["ANIO"] >= 2010].copy()
    d["fake_post_mmi"] = (d["ANIO"] >= 2014).astype(int) * d["mean_mmi"]
    m = smf.wls("_detr ~ fake_post_mmi + C(jr) + C(jt)", data=d, weights=d[W].clip(lower=1)).fit(
        cov_type="cluster", cov_kwds={"groups": d["REGION"]}
    )
    return {
        "check": label,
        "avg_beta": np.nan,
        "avg_se": np.nan,
        "avg_p": np.nan,
        "y2014_beta": m.params.get("fake_post_mmi", np.nan),
        "y2014_se": m.bse.get("fake_post_mmi", np.nan),
        "y2014_p": m.pvalues.get("fake_post_mmi", np.nan),
        "n": int(m.nobs),
    }


def se_fmt(x: float) -> str:
    return "" if pd.isna(x) else f"({x:.3f})"


def write_tex(summary: pd.DataFrame) -> None:
    """Three-panel table with standard errors, as it appears in the paper."""
    s = summary.set_index("check")
    labels = {
        "C1 composition: establishments": "Log establishments",
        "C1 composition: employment": "Log employment",
        "C1 composition: gross output": "Log gross output",
        "C2 placebo 2014 within post": "Fake 2014 shock within post-period",
        "C2 unskilled wage baseline": "Unskilled wage",
        "C4 skilled wage internal placebo": "Skilled wage",
        "C4 skill-premium effect": "Skill-premium effect",
        "C3 + region-year controls": "Unskilled wage + region-year controls",
        "C5 food only": "Food only",
        "C5 reconstruction-adjacent": "Wood/basic metals",
    }
    panels = [
        ("Panel A. Composition and placebo checks",
         ["C1 composition: establishments", "C1 composition: employment",
          "C1 composition: gross output", "C2 placebo 2014 within post"]),
        ("Panel B. Wage decomposition and internal placebo",
         ["C2 unskilled wage baseline", "C4 skilled wage internal placebo",
          "C4 skill-premium effect", "C3 + region-year controls"]),
        ("Panel C. Industry heterogeneity",
         ["C5 food only", "C5 reconstruction-adjacent"]),
    ]
    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Robustness of the wage mechanism: alternative explanations}",
        r"\label{tab:wage_mechanism_checks}",
        r"\begin{threeparttable}",
        r"\small\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & Avg.\ post $\times$ MMI & 2014 $\times$ MMI & $N$ \\",
        r"\midrule",
    ]
    for panel_i, (panel, keys) in enumerate(panels):
        if panel_i:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{panel}}}}} \\")
        for key in keys:
            r = s.loc[key]
            lines.append(rf"{labels[key]} & {fmt(r['avg_beta'], r['avg_p'])} & "
                         rf"{fmt(r['y2014_beta'], r['y2014_p'])} & {fmt_n(r['n'])} \\")
            lines.append(rf" & {se_fmt(r['avg_se'])} & {se_fmt(r['y2014_se'])} & \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]\footnotesize",
        r"\item \textit{Notes.} Coefficients are per unit of regional MMI after pre-period-projected detrending. "
        r"The wage rows use cell-mean log wages in the MAIN sample unless otherwise noted. "
        r"Panel A reports 2014 event-study coefficients for non-wage outcomes and a fake-shock placebo estimated only within the post-earthquake period. "
        r"Region-year controls are leave-one-industry-out ENIA manufacturing aggregates: log establishments, employment, output, investment, and regional production wage. "
        r"Standard errors, clustered by region, are in parentheses. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ])
    (PAPER_TABLES / "tab_wage_mechanism_checks.tex").write_text("\n".join(lines) + "\n")


def plot_wage_event(es: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {
        "C2 unskilled wage baseline": "darkorange",
        "C4 skilled wage internal placebo": "darkblue",
        "C4 skill-premium effect": "darkgreen",
    }
    labels = {
        "C2 unskilled wage baseline": "Unskilled wage",
        "C4 skilled wage internal placebo": "Skilled wage",
        "C4 skill-premium effect": "Skill-premium effect",
    }
    for key, color in colors.items():
        d = es[es["check"] == key].sort_values("year")
        ax.errorbar(d["year"], d["beta"], yerr=1.96 * d["se"], marker="o", capsize=3, color=color, label=labels[key], alpha=0.9)
    ax.axhline(0, color="gray", lw=0.7)
    ax.axvline(2009.5, color="black", ls="--", lw=1)
    ax.set_xlabel("year")
    ax.set_ylabel(r"$\beta$ (year $\times$ MMI)")
    ax.set_title("Wage mechanism check: unskilled wage vs internal placebos", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "E10_wage_mechanism_checks.png", dpi=140)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)

    wage = build_wage_cell()
    wage_ctrl = add_region_year_controls(wage)

    summary: list[dict] = []
    events: list[pd.DataFrame] = []

    # C1. Composition checks from the existing earthquake event-study output.
    es_inv = pd.read_csv(OUT / "earthquake_event_study_pretrend.csv")
    comp_map = {
        "log N_est (entry/exit)": "C1 composition: establishments",
        "log employment (EMPTOT)": "C1 composition: employment",
        "log gross output (VBP)*": "C1 composition: gross output",
    }
    for outcome, label in comp_map.items():
        d = es_inv[(es_inv["outcome"] == outcome) & (es_inv["year"] == 2014)].iloc[0]
        summary.append(
            {
                "check": label,
                "avg_beta": np.nan,
                "avg_se": np.nan,
                "avg_p": np.nan,
                "y2014_beta": d["beta"],
                "y2014_se": d["se"],
                "y2014_p": d["p"],
                "n": np.nan,
            }
        )

    # C2/C4. Wage event studies and average effects.
    for ycol, label in [
        ("log_wunsk", "C2 unskilled wage baseline"),
        ("log_wskill", "C4 skilled wage internal placebo"),
        ("log_skillprem", "C4 skill-premium effect"),
    ]:
        sub = detrend(wage, ycol, MAIN_IND2)
        events.append(estimate_event(sub, label))
        summary.append(estimate_avg_and_2014(sub, label))
        if ycol == "log_wunsk":
            summary.append(placebo_post_2014(sub, "C2 placebo 2014 within post"))

    # C3. Region-year manufacturing controls.
    controls = ["loo_log_n_est", "loo_log_emp", "loo_log_vbp", "loo_log_inv", "ry_log_wage"]
    sub_ctrl = detrend(wage_ctrl, "log_wunsk", MAIN_IND2)
    events.append(estimate_event(sub_ctrl, "C3 + region-year controls", controls=controls))
    summary.append(estimate_avg_and_2014(sub_ctrl, "C3 + region-year controls", controls=controls))

    # C5. Industry heterogeneity: food vs reconstruction-adjacent industries.
    for inds, label in [(["10"], "C5 food only"), (RECON_ADJ, "C5 reconstruction-adjacent")]:
        sub = detrend(wage, "log_wunsk", inds)
        events.append(estimate_event(sub, label))
        summary.append(estimate_avg_and_2014(sub, label))

    summary_df = pd.DataFrame(summary)
    event_df = pd.concat(events, ignore_index=True).sort_values(["check", "year"])

    summary_df.to_csv(OUT / "wage_mechanism_checks.csv", index=False)
    event_df.to_csv(OUT / "wage_mechanism_event_studies.csv", index=False)
    write_tex(summary_df)
    plot_wage_event(event_df)

    print("=== Wage mechanism checks ===")
    print(summary_df.round(4).to_string(index=False))
    print(f"\nWrote {OUT / 'wage_mechanism_checks.csv'}")
    print(f"Wrote {OUT / 'wage_mechanism_event_studies.csv'}")
    print(f"Wrote {PAPER_TABLES / 'tab_wage_mechanism_checks.tex'}")
    print(f"Wrote {FIGS / 'E10_wage_mechanism_checks.png'}")


if __name__ == "__main__":
    main()
