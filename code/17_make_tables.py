"""Generate applied-micro LaTeX (booktabs) tables + equations from the result CSVs.

Reproducible: re-run after any regression change and the .tex updates exactly.
Outputs (paper/):
  tables/tab_main_did.tex        Table 1 - main DiD (Post x MMI), no-trend vs +region trends
  tables/tab_event_study.tex     Table 2 - investment event-study coefficients (pre-projected)
  tables/tab_robustness.tex      Table 3 - robustness of the 2014 investment pulse
  equations.tex                  the specification equations
  paper.tex                      minimal standalone doc that \\input{}s the above
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
PAPER = ROOT / "paper"
TAB = PAPER / "tables"
TAB.mkdir(parents=True, exist_ok=True)
MAIN_IND2 = ["10", "16", "24"]
PRE = [2003, 2004, 2005, 2007, 2008, 2009]


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def cell(beta: float, se: float, p: float) -> tuple[str, str]:
    """Return (coef-with-stars, (se)) strings, or dashes if missing."""
    if pd.isna(beta):
        return "--", ""
    return f"{beta:.3f}{stars(p)}", f"({se:.3f})"


# ----------------------------------------------------------------------------
# Table 1: main DiD, columns = outcomes, Panel A (no trend) / Panel B (+region trends)
# ----------------------------------------------------------------------------
def table_main():
    df = pd.read_csv(OUT / "earthquake_did_main.csv")
    df = df[(df["sample"] == "MAIN (Food/Wood/Metals)") & (df["treat"] == "post_mmi")]
    outcomes = [("log investment (CBN/CB)", "Investment"),
                ("log employment (EMPTOT)", "Employment"),
                ("log wage (REGPRDI)", "Wage"),
                ("log N_est (entry/exit)", "Establishments"),
                ("log gross output (VBP)*", "Gross output")]
    ncol = len(outcomes)

    def panel(trends: str):
        sub = df[df["trends"] == trends].set_index("outcome")
        coefs, ses = [], []
        n = None
        for key, _ in outcomes:
            r = sub.loc[key] if key in sub.index else None
            if r is None:
                coefs.append("--"); ses.append(""); continue
            c, s = cell(r["beta"], r["se"], r["p"])
            coefs.append(c); ses.append(s)
            n = int(r["n"])
        return coefs, ses, n

    cA, sA, nA = panel("none")
    cB, sB, nB = panel("linear")
    cC, sC, nC = panel("quad")
    head = " & ".join(f"\\multicolumn{{1}}{{c}}{{{lab}}}" for _, lab in outcomes)
    colspec = "l" + "c" * ncol

    L = []
    L.append(r"\begin{table}[htbp]\centering")
    L.append(r"\caption{Earthquake exposure and manufacturing adjustment, 2003--2017}")
    L.append(r"\label{tab:main_did}")
    L.append(r"\begin{threeparttable}")
    L.append(r"\small\setlength{\tabcolsep}{4pt}")
    L.append(rf"\begin{{tabular}}{{{colspec}}}")
    L.append(r"\toprule")
    L.append(" & " + head + r" \\")
    L.append(f"\\cmidrule(lr){{2-{ncol+1}}}")
    L.append(" & " + " & ".join(f"({i+1})" for i in range(ncol)) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{%d}{l}{\textit{Panel A. No region trends}} \\" % (ncol + 1))
    L.append(r"$\text{Post}_t \times \text{MMI}_r$ & " + " & ".join(cA) + r" \\")
    L.append(" & " + " & ".join(sA) + r" \\")
    L.append(rf"Observations & \multicolumn{{{ncol}}}{{c}}{{{nA}}} \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{%d}{l}{\textit{Panel B. + Region linear trends}} \\" % (ncol + 1))
    L.append(r"$\text{Post}_t \times \text{MMI}_r$ & " + " & ".join(cB) + r" \\")
    L.append(" & " + " & ".join(sB) + r" \\")
    L.append(rf"Observations & \multicolumn{{{ncol}}}{{c}}{{{nB}}} \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{%d}{l}{\textit{Panel C. + Region quadratic trends}} \\" % (ncol + 1))
    L.append(r"$\text{Post}_t \times \text{MMI}_r$ & " + " & ".join(cC) + r" \\")
    L.append(" & " + " & ".join(sC) + r" \\")
    L.append(rf"Observations & \multicolumn{{{ncol}}}{{c}}{{{nC}}} \\")
    L.append(r"\midrule")
    L.append(r"Industry $\times$ region FE & " + " & ".join([r"\checkmark"] * ncol) + r" \\")
    L.append(r"Industry $\times$ year FE & " + " & ".join([r"\checkmark"] * ncol) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\begin{tablenotes}[flushleft]\footnotesize")
    L.append(r"\item \textit{Notes.} Cells are 2-digit ISIC industry $\times$ region $\times$ year, "
             r"sample restricted to Food (10), Wood (16) and Basic metals (24). "
             r"$\text{MMI}_r$ is the region's area-weighted Modified Mercalli Intensity from the "
             r"2010 Maule earthquake (USGS ShakeMap); $\text{Post}_t=\mathbf{1}[t\geq 2010]$. "
             r"Each entry is a separate WLS regression weighted by cell effective $N$; standard "
             r"errors clustered by region (15 clusters) in parentheses. Panels B and C add "
             r"full-sample region-specific linear and quadratic time trends; the quadratic panel is "
             r"a conventional nonlinear-trend robustness, while the principled sensitivity to "
             r"parallel-trends violations is the HonestDiD analysis in Section~\ref{sec:honest}. "
             r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.")
    L.append(r"\end{tablenotes}")
    L.append(r"\end{threeparttable}")
    L.append(r"\end{table}")
    (TAB / "tab_main_did.tex").write_text("\n".join(L))
    print("wrote tables/tab_main_did.tex")


# ----------------------------------------------------------------------------
# Table 2: investment event-study coefficients (pre-period-projected detrending)
# ----------------------------------------------------------------------------
def table_event_study():
    df = pd.read_csv(OUT / "earthquake_event_study_pretrend.csv")
    df = df[df["outcome"] == "log investment (CBN/CB)"].sort_values("year")
    L = []
    L.append(r"\begin{table}[htbp]\centering")
    L.append(r"\caption{Investment event study: $\beta_k$ on $\mathbf{1}[t=k]\times\text{MMI}_r$ "
             r"(reference year 2009)}")
    L.append(r"\label{tab:event_study}")
    L.append(r"\begin{threeparttable}")
    L.append(r"\begin{tabular}{lcc}")
    L.append(r"\toprule")
    L.append(r"Year $k$ & $\hat\beta_k$ & (s.e.) \\")
    L.append(r"\midrule")
    for _, r in df.iterrows():
        yr = int(r["year"])
        if yr == 2009:
            L.append(r"2009 & \multicolumn{2}{c}{--- reference ---} \\")
            continue
        c, s = cell(r["beta"], r["se"], r["p"])
        tag = r"\quad\textit{(earthquake)}" if yr == 2010 else ""
        L.append(rf"{yr}{tag} & {c} & {s} \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\begin{tablenotes}[flushleft]\footnotesize")
    L.append(r"\item \textit{Notes.} MAIN sample (Food/Wood/Metals). Region linear trends are "
             r"estimated on the pre-earthquake years (2003--2009) only and projected forward, so "
             r"the post-quake response does not contaminate the trend. Industry$\times$region and "
             r"industry$\times$year fixed effects included; SE clustered by region. "
             r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.")
    L.append(r"\end{tablenotes}")
    L.append(r"\end{threeparttable}")
    L.append(r"\end{table}")
    (TAB / "tab_event_study.tex").write_text("\n".join(L))
    print("wrote tables/tab_event_study.tex")


# ----------------------------------------------------------------------------
# Table 3: robustness of the 2014 investment pulse
# ----------------------------------------------------------------------------
def table_robustness():
    df = pd.read_csv(OUT / "earthquake_robustness_pulse.csv")
    L = []
    L.append(r"\begin{table}[htbp]\centering")
    L.append(r"\caption{Robustness of the 2014 reconstruction-investment pulse}")
    L.append(r"\label{tab:robustness}")
    L.append(r"\begin{threeparttable}")
    L.append(r"\begin{tabular}{lccc}")
    L.append(r"\toprule")
    L.append(r"Specification & 2014 $\hat\beta$ & 2015 $\hat\beta$ & $N$ \\")
    L.append(r"\midrule")
    for _, r in df.iterrows():
        name = (r["variant"].replace("&", r"\&")
                .replace(">=", r"$\geq$").replace("x", r"$\times$"))
        b14 = f"{r['beta_2014']:.3f}{stars(r['p_2014'])}"
        se14 = f"({r['se_2014']:.3f})"
        b15 = f"{r['beta_2015']:.3f}{stars(r['p_2015'])}"
        L.append(rf"{name} & {b14} & {b15} & {int(r['n'])} \\")
        L.append(rf" & {se14} & & \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\begin{tablenotes}[flushleft]\footnotesize")
    wcb = pd.read_csv(OUT / "wild_bootstrap.csv")
    pinv = wcb.loc[wcb["estimate"].str.contains("investment"), "p_wcb"]
    wcb_note = (rf" The 2014 pulse survives a restricted wild cluster bootstrap "
                rf"(Rademacher, $B=999$): $p={float(pinv.iloc[0]):.3f}$." if len(pinv) else "")
    L.append(r"\item \textit{Notes.} Each row re-estimates the 2014 (and 2015) coefficient on "
             r"$\mathbf{1}[t=k]\times\text{MMI}_r$ for log investment under one design change, using "
             r"pre-period-projected detrending. SE (2014) clustered by region in parentheses." + wcb_note +
             r" $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.")
    L.append(r"\end{tablenotes}")
    L.append(r"\end{threeparttable}")
    L.append(r"\end{table}")
    (TAB / "tab_robustness.tex").write_text("\n".join(L))
    print("wrote tables/tab_robustness.tex")


# ----------------------------------------------------------------------------
# Table 0: summary statistics (pre-period means by high vs low MMI)
# ----------------------------------------------------------------------------
def table_summary():
    q = pd.read_parquet(DATA / "enia_cell_2digit_quake.parquet")
    lab = pd.read_parquet(DATA / "enia_cell_labor.parquet")
    q = q[q["ind2"].isin(MAIN_IND2) & q["ANIO"].isin(PRE)].copy()
    lab = lab[lab["ind2"].isin(MAIN_IND2) & lab["ANIO"].isin(PRE)].copy()
    q["high"] = q["mean_mmi"] >= 6.0
    lab["high"] = lab["mean_mmi"] >= 6.0

    def wm(df, col, hi):
        d = df[(df["high"] == hi)].dropna(subset=[col])
        w = d["weight_sum"].clip(lower=1)
        return (d[col] * w).sum() / w.sum() if len(d) else float("nan")

    rows = [("Mean MMI", "mean_mmi", q, False),
            ("Investment (CLP, cell sum)", "inv", q, False),
            ("Employment (cell sum)", "emp", q, False),
            ("Gross output VBP (cell sum)", "vbp", q, False),
            ("Log skill premium", "log_skillprem", lab, False)]
    L = [r"\begin{table}[htbp]\centering",
         r"\caption{Summary statistics, pre-earthquake cells (2003--2009)}",
         r"\label{tab:summary}", r"\begin{threeparttable}",
         r"\begin{tabular}{lcc}", r"\toprule",
         r" & High MMI ($\geq 6$) & Low MMI ($<6$) \\", r"\midrule"]
    for name, col, df, _ in rows:
        hi, lo = wm(df, col, True), wm(df, col, False)
        fmt = (lambda x: f"{x:,.0f}") if col in ("inv", "emp", "vbp") else (lambda x: f"{x:.3f}")
        L.append(rf"{name} & {fmt(hi)} & {fmt(lo)} \\")
    nhi = q[q["high"]].shape[0]; nlo = q[~q["high"]].shape[0]
    L.append(r"\midrule")
    L.append(rf"Cell-years & {nhi} & {nlo} \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{tablenotes}[flushleft]\footnotesize",
          r"\item \textit{Notes.} Weighted means across industry$\times$region$\times$year cells "
          r"(Food, Wood, Basic metals) over 2003--2009. High-MMI regions are those with "
          r"area-weighted intensity $\geq 6$.",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    (TAB / "tab_summary.tex").write_text("\n".join(L))
    print("wrote tables/tab_summary.tex")


# ----------------------------------------------------------------------------
# Table 4: labor margin -- skill share, skill premium, wage decomposition
# ----------------------------------------------------------------------------
def table_labor():
    """Wage-based skill premium: the canonical capital-skill test. Uses clearly
    labelled occupational WAGE fields (tecnicos/especialistas vs no calificado);
    the occupational EMPLOYMENT grid is not used (its categories do not map cleanly
    to skill). All coefficients are avg. post-2010 deviation from the pre-trend."""
    wd = pd.read_csv(OUT / "wage_decomposition_summary.csv").set_index("wage")
    sk = wd.loc["Skilled wage (REGTESP)"]
    un = wd.loc["Unskilled wage (REGNOCALD)"]
    prem_b = sk["avg_post"] - un["avg_post"]   # log premium change = skilled - unskilled

    L = [r"\begin{table}[htbp]\centering",
         r"\caption{The labor margin: reconstruction raises low-skill wages}",
         r"\label{tab:labor}", r"\begin{threeparttable}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r" & Skilled wage & Unskilled wage & Skill premium \\",
         r" & (1) & (2) & (3)$=$(1)$-$(2) \\", r"\midrule",
         rf"Avg.\ post-2010 $\times$ MMI & {sk['avg_post']:.3f}{stars(sk['p'])} & "
         rf"{un['avg_post']:.3f}{stars(un['p'])} & {prem_b:.3f} \\",
         r"\midrule",
         r"\multicolumn{4}{l}{\textit{By sub-period ($\times$ MMI):}} \\",
         rf"\quad 2010--2014 & {sk['early_2010_14']:.3f}{stars(sk['early_p'])} & "
         rf"{un['early_2010_14']:.3f}{stars(un['early_p'])} & "
         rf"{sk['early_2010_14']-un['early_2010_14']:.3f} \\",
         rf"\quad 2015--2016 & {sk['late_2015_16']:.3f}{stars(sk['late_p'])} & "
         rf"{un['late_2015_16']:.3f}{stars(un['late_p'])} & "
         rf"{sk['late_2015_16']-un['late_2015_16']:.3f} \\",
         r"\bottomrule", r"\end{tabular}",
         r"\begin{tablenotes}[flushleft]\footnotesize",
         r"\item \textit{Notes.} Skilled wage $=$ REGTESP (technicians/specialists), unskilled $=$ "
         r"REGNOCALD (no calificado), 2003--2016. Each coefficient is the avg.\ post-2010 "
         r"deviation of the cell-mean log wage from its pre-period-projected region trend, per unit "
         r"MMI; MAIN sample; SE clustered by region. The skill premium falls because the "
         r"\emph{unskilled} wage rises (peaking in 2014, Fig.~\ref{fig:wagedecomp}), not because the "
         r"skilled wage falls --- the canonical capital--skill complementarity prediction (a rising "
         r"premium) is rejected. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
         r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    (TAB / "tab_labor.tex").write_text("\n".join(L))
    print("wrote tables/tab_labor.tex")


# ----------------------------------------------------------------------------
# Table 5: mechanism -- asset composition, net accumulation, wage decomposition
# ----------------------------------------------------------------------------
def table_mechanism():
    asset = pd.read_csv(OUT / "asset_decomposition_pulse.csv")
    net = pd.read_csv(OUT / "net_investment_test.csv")
    wd = pd.read_csv(OUT / "wage_decomposition_summary.csv")

    def a14(name):
        r = asset[(asset["asset"].str.startswith(name)) & (asset["year"] == 2014)]
        return cell(r["beta"].iloc[0], r["se"].iloc[0], r["p"].iloc[0])

    L = [r"\begin{table}[htbp]\centering",
         r"\caption{Mechanism: reconstruction is low-skill-labor intensive}",
         r"\label{tab:mechanism}", r"\begin{threeparttable}",
         r"\begin{tabular}{lcc}", r"\toprule",
         r"\multicolumn{3}{l}{\textit{Panel A. 2014 investment pulse by asset class ($\times$ MMI)}} \\",
         r" & $\hat\beta$ & (s.e.) \\", r"\midrule"]
    for nm, lab in [("Structures", "Structures"), ("Machinery", "Machinery"), ("Equipment", "Equipment")]:
        c, s = a14(nm)
        L.append(rf"{lab} & {c} & {s} \\")
    L += [r"\midrule",
          r"\multicolumn{3}{l}{\textit{Panel B. Net accumulation: avg.\ post-2010 deviation ($\times$ MMI)}} \\",
          r" & 2010--14 & 2015--17 \\", r"\midrule"]
    for _, r in net.iterrows():
        e = f"{r['early_2010_14']:.3f}{stars(r['early_p'])}"
        la = f"{r['late_2015_17']:.3f}{stars(r['late_p'])}"
        L.append(rf"{r['capital']} & {e} & {la} \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{tablenotes}[flushleft]\footnotesize",
          r"\item \textit{Notes.} From the pre-period-projected detrended residual, MAIN sample, "
          r"SE clustered by region. Panel A: the 2014 pulse spans both structures and equipment. "
          r"Panel B: equipment accumulation persists (net stock rises), while structures is "
          r"transient. So the absence of a skill response (Table~\ref{tab:labor}) is not because no "
          r"equipment was bought. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    (TAB / "tab_mechanism.tex").write_text("\n".join(L))
    print("wrote tables/tab_mechanism.tex")


# ----------------------------------------------------------------------------
# Table: incidence decomposition -- where does the reconstruction shock show up?
# ----------------------------------------------------------------------------
def table_incidence():
    df = pd.read_csv(OUT / "incidence_did.csv")
    L = [r"\begin{table}[htbp]\centering",
         r"\caption{Where the reconstruction shock shows up: incidence decomposition}",
         r"\label{tab:incidence}", r"\begin{threeparttable}",
         r"\small", r"\begin{tabular}{lcc}", r"\toprule",
         r"Outcome (cell, logs) & $\text{Post}\times\text{MMI}$ & (s.e.) \\", r"\midrule"]
    for _, r in df.iterrows():
        c, s = cell(r["beta"], r["se"], r["p"])
        L.append(rf"{r['outcome']} & {c} & {s} \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\begin{tablenotes}[flushleft]\footnotesize")
    L.append(r"\item \textit{Notes.} Each row is a separate regression of the avg.\ post-2010 "
             r"deviation from the region's pre-period-projected trend on $\text{Post}\times\text{MMI}$, "
             r"MAIN sample, SE clustered by region. Only investment (capital formation) responds; "
             r"output, employment, labor productivity, the labor share of output, and the profit "
             r"proxy do not---the rebuilding restored capital without changing the scale or factor "
             r"distribution of production. The wage bill drifts up but is fragile (see "
             r"Section~\ref{sec:honest}); the labor share is flat. Value-added-based labor shares "
             r"are infeasible (value added is recorded only in 2003--07 and 2014--17); RENIMP is a "
             r"noisy profit proxy (it nets investment-linked depreciation and the 2014 rate change, "
             r"the latter absorbed by industry$\times$year FE). "
             r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.")
    L += [r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    (TAB / "tab_incidence.tex").write_text("\n".join(L))
    print("wrote tables/tab_incidence.tex")


def table_spillover():
    df = pd.read_csv(OUT / "spillover_test.csv")
    L = [r"\begin{table}[htbp]\centering",
         r"\caption{Spatial-lag (SLX) spillover test: own versus neighbor exposure}",
         r"\label{tab:spillover}", r"\begin{threeparttable}",
         r"\small", r"\begin{tabular}{lccc}", r"\toprule",
         r" & \multicolumn{2}{c}{Own $\text{Post}\times\text{MMI}$} "
         r"& Neighbor $\text{Post}\times\text{MMI}$ \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}",
         r"Outcome (cell, logs) & Baseline & + Neighbor & (augmented) \\", r"\midrule"]
    for _, r in df.iterrows():
        bo, so = cell(r["beta_own_only"], r["se_own_only"], r["p_own_only"])
        ba, sa = cell(r["beta_own_aug"], r["se_own_aug"], r["p_own_aug"])
        bn, sn = cell(r["theta_nbr"], r["se_nbr"], r["p_nbr"])
        L.append(rf"{r['outcome']} & {bo} & {ba} & {bn} \\")
        L.append(rf" & {so} & {sa} & {sn} \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\begin{tablenotes}[flushleft]\footnotesize")
    L.append(r"\item \textit{Notes.} ``Neighbor exposure'' is the mean MMI of a region's "
             r"immediate north--south geographic neighbors (Chile is effectively "
             r"one-dimensional). Each outcome is regressed, on the pre-period-projected-detrended "
             r"footing of Table~\ref{tab:incidence} (MAIN sample, SE clustered by region), first "
             r"on own $\text{Post}\times\text{MMI}$ alone (``Baseline''), then on own and neighbor "
             r"$\text{Post}\times\text{MMI}$ jointly (``+ Neighbor''/``augmented''). Own and "
             r"neighbor MMI are collinear across regions ($r\approx0.98$), so the two terms are "
             r"not separately identified: the augmented own-coefficients are unstable (the output "
             r"own-coefficient inflates with an opposite-signed neighbor term) and standard errors "
             r"are large (the investment neighbor coefficient has a standard error above one). The "
             r"sign of any cross-region spillover is therefore not resolvable with this design. "
             r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.")
    L += [r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    (TAB / "tab_spillover.tex").write_text("\n".join(L))
    print("wrote tables/tab_spillover.tex")


# ----------------------------------------------------------------------------
# Table: HonestDiD sensitivity (Rambachan & Roth 2023) for the 2014 effects
# ----------------------------------------------------------------------------
def table_honestdid():
    df = pd.read_csv(OUT / "honestdid_results.csv")

    def ci(outcome, restriction, param=None):
        d = df[(df["outcome"] == outcome) & (df["restriction"] == restriction)]
        if param is not None:
            d = d[np.isclose(d["param"], param)]
        if d.empty or pd.isna(d["lb"].iloc[0]):
            return "--"
        lb, ub = d["lb"].iloc[0], d["ub"].iloc[0]
        excl = (lb > 0) or (ub < 0)
        s = f"[{lb:.2f},\\,{ub:.2f}]"
        return (r"\textbf{" + s + r"}") if excl else s

    def bd(outcome, restriction):
        d = df[(df["outcome"] == outcome) & (df["restriction"] == restriction)]
        return f"{d['param'].iloc[0]:.2f}" if not d.empty else "--"

    cols = ["Investment", "Unskilled wage"]
    L = [r"\begin{table}[htbp]\centering",
         r"\caption{HonestDiD sensitivity of the 2014 effect to parallel-trends violations}",
         r"\label{tab:honestdid}", r"\begin{threeparttable}",
         r"\begin{tabular}{lcc}", r"\toprule",
         r" & Investment & Unskilled wage \\", r"\midrule",
         r"Original CI (PT imposed) & " + " & ".join(ci(c, "Original (PT)") for c in cols) + r" \\",
         r"\addlinespace",
         r"\multicolumn{3}{l}{\textit{Relative magnitudes} $\Delta^{RM}(\bar M)$:} \\",
         r"\quad $\bar M=0.5$ & " + " & ".join(ci(c, "RM", 0.5) for c in cols) + r" \\",
         r"\quad $\bar M=1.0$ & " + " & ".join(ci(c, "RM", 1.0) for c in cols) + r" \\",
         r"\quad breakdown $\bar M$ & " + " & ".join(bd(c, "breakdown_Mbar") for c in cols) + r" \\",
         r"\addlinespace",
         r"\multicolumn{3}{l}{\textit{Smoothness} $\Delta^{SD}(M)$:} \\",
         r"\quad $M=0$ (linear trend) & " + " & ".join(ci(c, "SD", 0.0) for c in cols) + r" \\",
         r"\quad $M=0.05$ & " + " & ".join(ci(c, "SD", 0.05) for c in cols) + r" \\",
         r"\quad breakdown $M$ & " + " & ".join(bd(c, "breakdown_M") for c in cols) + r" \\",
         r"\addlinespace",
         r"Spike contrast, original CI & " + " & ".join(ci(c, "spike_original") for c in cols) + r" \\",
         r"\bottomrule", r"\end{tabular}",
         r"\begin{tablenotes}[flushleft]\footnotesize",
         r"\item \textit{Notes.} Robust 95\% confidence sets for the 2014 (year $\times$ MMI) "
         r"coefficient from the raw event study, following \citet{rambachanroth2023}. "
         r"$\Delta^{RM}(\bar M)$ bounds the post-period violation of parallel trends by $\bar M$ "
         r"times the largest pre-period violation; $\Delta^{SD}(M)$ bounds the change in the "
         r"differential trend's slope by $M$ per period ($M=0$ is exactly linear, equivalent to the "
         r"paper's pre-period detrending). Breakdown is the value at which the robust CI first "
         r"includes zero. \textbf{Bold} intervals exclude zero. The ``spike contrast'' targets "
         r"$\beta_{2014}-\tfrac12(\beta_{2013}+\beta_{2015})$, which nets out a locally smooth trend "
         r"but also the multi-year pulse, and is uninformative here.",
         r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    (TAB / "tab_honestdid.tex").write_text("\n".join(L))
    print("wrote tables/tab_honestdid.tex")


# ----------------------------------------------------------------------------
# Table: CASEN out-migration check (regional population by skill)
# ----------------------------------------------------------------------------
def table_casen():
    did = pd.read_csv(OUT / "casen_migration_did.csv").set_index("outcome")
    es = pd.read_csv(OUT / "casen_migration_event_study.csv")
    cols = [("Unskilled population", "Unskilled pop."),
            ("Skilled population", "Skilled pop."),
            ("Total population", "Total pop."),
            ("Unskilled share", "Unsk.\\ share")]

    def coef(name):
        r = did.loc[name]
        return cell(r["beta"], r["se"], r["p"])

    # joint pre-trend test + HonestDiD robust CI (unskilled population)
    hdf = pd.read_csv(OUT / "casen_honestdid.csv")
    fr = hdf[hdf["kind"] == "pretrend_F"].iloc[0]
    ftxt = f"$F={fr['param']:.2f}$ ($p={fr['p']:.2f}$)"
    rm1 = hdf[(hdf["kind"] == "RM") & np.isclose(hdf["param"], 1.0)].iloc[0]
    rmtxt = f"$[{rm1['lb']:.2f},\\,{rm1['ub']:.2f}]$"

    head = " & ".join(f"\\multicolumn{{1}}{{c}}{{{lab}}}" for _, lab in cols)
    n = int(did["n"].iloc[0])
    L = [r"\begin{table}[htbp]\centering",
         r"\caption{Out-migration check: regional population by skill (CASEN, 2003--2017)}",
         r"\label{tab:casen}", r"\begin{threeparttable}",
         r"\small\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         " & " + head + r" \\",
         r"\cmidrule(lr){2-5}",
         " & " + " & ".join(f"({i+1})" for i in range(4)) + r" \\", r"\midrule"]
    coefs = [coef(c) for c, _ in cols]
    L.append(r"$\text{Post}_t \times \text{MMI}_r$ & " + " & ".join(c for c, _ in coefs) + r" \\")
    L.append(" & " + " & ".join(s for _, s in coefs) + r" \\")
    L.append(r"\midrule")
    L.append(rf"Joint pre-trend test, unsk. & \multicolumn{{4}}{{l}}{{{ftxt}}} \\")
    L.append(rf"HonestDiD robust CI ($\bar M=1$) & \multicolumn{{4}}{{l}}{{{rmtxt}}} \\")
    L.append(rf"Region $\times$ year cells & \multicolumn{{4}}{{c}}{{{n}}} \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{tablenotes}[flushleft]\footnotesize",
          r"\item \textit{Notes.} Cells are region $\times$ CASEN-wave; outcomes are the log of the "
          r"population-weighted count of working-age (15--64) persons in each group, except column (4) "
          r"(the unskilled share). Skill is defined by education: unskilled $=$ basic schooling or "
          r"less, skilled $=$ tertiary. $\text{Post}_t=\mathbf{1}[t\geq 2010]$ (the 2009 wave is "
          r"pre-earthquake; the first post-earthquake wave is 2011). Regions are harmonized to the "
          r"pre-2007 13-region definition; SE clustered by region. A negative coefficient on column "
          r"(1) would indicate unskilled out-migration from harder-hit regions; the coefficient is "
          r"instead near zero. The joint pre-trend test (the 2003 and 2006 event-study coefficients "
          r"equal zero) is not rejected, and the HonestDiD relative-magnitudes robust CI for the "
          r"average post-earthquake effect at $\bar M=1$ \citep{rambachanroth2023} still rules out "
          r"any large unskilled out-migration. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    (TAB / "tab_casen.tex").write_text("\n".join(L))
    print("wrote tables/tab_casen.tex")


# ----------------------------------------------------------------------------
# Equations + standalone skeleton
# ----------------------------------------------------------------------------
def equations_and_skeleton():
    eqs = r"""% Specification equations for the 2010-earthquake / ENIA design.

% --- Main difference-in-differences ---
\begin{equation}
  y_{jrt} \;=\; \beta \,\bigl(\text{Post}_t \times \text{MMI}_r\bigr)
  \;+\; \gamma_{jr} \;+\; \delta_{jt} \;+\; \varepsilon_{jrt},
  \label{eq:did}
\end{equation}
% where j = 2-digit ISIC industry, r = region, t = year; Post_t = 1[t >= 2010];
% MMI_r = region area-weighted Modified Mercalli Intensity of the 2010 Maule
% earthquake; gamma_{jr} = industry x region FE; delta_{jt} = industry x year FE.

% --- Event study (relative to reference year 2009) ---
\begin{equation}
  y_{jrt} \;=\; \sum_{k \neq 2009} \beta_k \,\bigl(\mathbf{1}[t=k]\times \text{MMI}_r\bigr)
  \;+\; \gamma_{jr} \;+\; \delta_{jt} \;+\; \varepsilon_{jrt}.
  \label{eq:eventstudy}
\end{equation}

% --- Pre-trend-projected detrending (preferred) ---
% Region linear trends theta_r are estimated on the pre-quake window only,
% y_{jrt} = a_{jr} + theta_r (t - 2009) + u_{jrt}  for t in {2003,...,2009},
% then projected forward; the event study (eq. \ref{eq:eventstudy}) is run on the
% residual y_{jrt} - \hat a_{jr} - \hat\theta_r (t-2009).
"""
    (PAPER / "equations.tex").write_text(eqs)
    print("wrote equations.tex")

    skel = r"""\documentclass[11pt]{article}
\usepackage{amsmath, booktabs, threeparttable, graphicx, geometry}
\usepackage{amssymb}                 % \checkmark, \geq
\geometry{margin=1in}
\title{The 2010 Maule Earthquake and Manufacturing Adjustment in Chile}
\author{Emily Liu}
\date{\today}

\begin{document}
\maketitle

\section{Empirical strategy}
I use the 2010 Maule earthquake ($M_w$ 8.8) as an exogenous regional capital
shock and estimate, on ENIA establishment data aggregated to industry $\times$
region $\times$ year cells,
\input{equations}      % equations \ref{eq:did} and \ref{eq:eventstudy}

\section{Results}
\input{tables/tab_main_did}
\input{tables/tab_event_study}

\section{Robustness}
\input{tables/tab_robustness}

\begin{figure}[htbp]\centering
  \includegraphics[width=\linewidth]{../output/figs/E5_event_study_pretrend.png}
  \caption{Event study with pre-period-projected region trends. The 2014--2015
  investment spike is a sharp deviation from the (flat) pre-trend.}
  \label{fig:eventstudy}
\end{figure}

\end{document}
"""
    # do NOT overwrite the full hand-written paper.tex if it already exists
    target = PAPER / ("paper.tex" if not (PAPER / "paper.tex").exists() else "paper_skeleton.tex")
    target.write_text(skel)
    print(f"wrote {target.name}")


if __name__ == "__main__":
    table_summary()
    table_main()
    table_event_study()
    table_robustness()
    table_labor()
    table_mechanism()
    table_incidence()
    table_spillover()
    table_honestdid()
    table_casen()
    equations_and_skeleton()
    print(f"\nAll LaTeX written under {PAPER}")
