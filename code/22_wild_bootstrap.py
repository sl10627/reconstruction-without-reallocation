"""Wild cluster bootstrap-t (Cameron-Gelbach-Miller) for the headline coefficients.

With only 15 region clusters, cluster-robust SEs are downward biased and the
asymptotic t-test over-rejects. I report restricted wild cluster bootstrap p-values
(Rademacher weights, B=999, null imposed) for:
  - 2014 investment x MMI pulse              (event study, focal = year-2014 term)
  - 2014 unskilled-wage x MMI                (event study, focal = year-2014 term)
  - equipment average post-2010 deviation    (net accumulation, focal = post x MMI)

Implemented in numpy on the pre-period-projected detrended residual with
industry x year (jt) fixed effects, weighted by cell effective size.

Output: output/wild_bootstrap.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "output"
MAIN_IND2 = ["10", "16", "24"]
MMI_FLOOR, W = 2.0, "weight_sum"
PRE_YEARS = [2003, 2004, 2005, 2006, 2007, 2008, 2009]
B = 999
SEED = 20100227  # earthquake date; fixed for reproducibility


def _sum(df, cols):
    cc = [c for c in cols if c in df.columns]
    return df[cc].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1) if cc else pd.Series(np.nan, index=df.index)


def build_pooled_cell(value_builder):
    """Aggregate establishment values to industry x region x year via value_builder(p)->Series."""
    p = pd.read_parquet(DATA / "enia_pooled.parquet")
    p = p[p["ind_rev4"].notna() & p["REGION"].notna() & p["ANIO"].notna()].copy()
    p["ind2"] = p["ind_rev4"].astype(str).str[:2]
    p["REGION"] = pd.to_numeric(p["REGION"], errors="coerce")
    p["ANIO"] = pd.to_numeric(p["ANIO"], errors="coerce").astype(int)
    p["w"] = pd.to_numeric(p["ind_weight"], errors="coerce").fillna(0.0)
    p["_val"], how = value_builder(p)
    rows = []
    for (i2, reg, yr), g in p.groupby(["ind2", "REGION", "ANIO"]):
        v, w = g["_val"], g["w"]
        if how == "sum":
            m = pd.notna(v) & (w > 0); agg = (v[m] * w[m]).sum() if m.any() else np.nan
        else:  # mean over positive
            m = v.notna() & (v > 0) & (w > 0); agg = (v[m] * w[m]).sum() / w[m].sum() if m.any() else np.nan
        rows.append({"ind2": i2, "REGION": int(reg), "ANIO": int(yr), "weight_sum": w.sum(), "val": agg})
    c = pd.DataFrame(rows)
    sh = pd.read_csv(DATA / "shakemap_by_region.csv").rename(columns={"enia_region": "REGION"})
    c = c.merge(sh[["REGION", "mean_mmi"]], on="REGION", how="left", validate="m:1")
    c["mean_mmi"] = c["mean_mmi"].fillna(MMI_FLOOR)
    c["year_c"] = c["ANIO"] - 2009
    c["jr"] = c["ind2"] + "_r" + c["REGION"].astype(str)
    c["jt"] = c["ind2"] + "_t" + c["ANIO"].astype(str)
    c["logval"] = np.where(c["val"] > 0, np.log(c["val"]), np.nan)
    return c


def detrend(cell):
    sub = cell[cell["ind2"].isin(MAIN_IND2)].dropna(subset=["logval"]).copy()
    for k in ["jr", "jt"]:
        vc = sub[k].value_counts(); sub = sub[sub[k].isin(vc[vc >= 2].index)]
    pre = sub[sub["ANIO"].isin(PRE_YEARS)]
    jr_mean = pre.groupby("jr").apply(lambda d: (d["logval"] * d[W]).sum() / d[W].sum(),
                                      include_groups=False).to_dict()
    sub = sub[sub["jr"].isin(jr_mean)].copy()
    sub["_dm"] = sub["logval"] - sub["jr"].map(jr_mean)
    slopes = {}
    for r, d in sub[sub["ANIO"].isin(PRE_YEARS)].groupby("REGION"):
        if d["year_c"].nunique() >= 2 and len(d) >= 3:
            X = np.c_[np.ones(len(d)), d["year_c"].values]
            wv = d[W].clip(lower=1).values
            b = np.linalg.lstsq(X * np.sqrt(wv)[:, None], d["_dm"].values * np.sqrt(wv), rcond=None)[0]
            slopes[r] = b[1]
        else:
            slopes[r] = 0.0
    sub["_detr"] = sub["_dm"] - sub["REGION"].map(slopes).fillna(0.0) * sub["year_c"]
    return sub


def _wls_cr(X, y, w, groups, focal):
    """Weighted OLS; return beta_focal, cluster-robust SE (CR1), t."""
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    XtX = Xw.T @ Xw
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (Xw.T @ yw)
    resid = y - X @ beta
    # cluster-robust meat: sum_g (X_g' W_g u_g)(.)'
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        idx = groups == g
        sg = (X[idx] * w[idx][:, None]).T @ resid[idx]
        meat += np.outer(sg, sg)
    G = len(np.unique(groups))
    n, k = X.shape
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    V = XtX_inv @ meat @ XtX_inv * adj
    se = np.sqrt(V[focal, focal])
    return beta[focal], se, beta, resid


def wild_cluster_boot(sub, treat_cols, focal_name, B=999, seed=SEED):
    """Restricted WCB-t p-value for coefficient focal_name in
    _detr ~ [treat_cols] + C(jt), weighted, clustered by REGION."""
    rng = np.random.default_rng(seed)
    jt = pd.get_dummies(sub["jt"], prefix="jt", drop_first=True).astype(float)
    Xfull = pd.concat([pd.Series(1.0, index=sub.index, name="const"),
                       sub[treat_cols].astype(float), jt], axis=1)
    cols = list(Xfull.columns)
    focal = cols.index(focal_name)
    X = Xfull.values
    y = sub["_detr"].values.astype(float)
    w = sub[W].clip(lower=1).values.astype(float)
    groups = sub["REGION"].values

    b_hat, se_hat, _, _ = _wls_cr(X, y, w, groups, focal)
    t_hat = b_hat / se_hat

    # restricted model: impose focal = 0 (drop focal column)
    keep = [i for i in range(X.shape[1]) if i != focal]
    Xr = X[:, keep]
    sw = np.sqrt(w)
    br = np.linalg.lstsq((Xr * sw[:, None]), y * sw, rcond=None)[0]
    yhat_r = Xr @ br
    ur = y - yhat_r

    uniq = np.unique(groups)
    count = 0
    for _ in range(B):
        wg = rng.choice([-1.0, 1.0], size=len(uniq))
        wmap = dict(zip(uniq, wg))
        wstar = np.array([wmap[g] for g in groups])
        ystar = yhat_r + wstar * ur
        b_s, se_s, _, _ = _wls_cr(X, ystar, w, groups, focal)
        t_s = b_s / se_s if se_s > 0 else 0.0
        if abs(t_s) >= abs(t_hat):
            count += 1
    p_wcb = (1 + count) / (B + 1)
    return b_hat, se_hat, t_hat, p_wcb


def main():
    results = []

    # ---- 1) 2014 investment pulse (event study, focal = 2014) ----
    inv = pd.read_parquet(DATA / "enia_cell_2digit_quake.parquet")
    inv = inv.rename(columns={"log_inv": "logval"})
    sub = detrend(inv)
    yrs = [y for y in [2003, 2004, 2005, 2006, 2007, 2008, 2010, 2011, 2013, 2014, 2015, 2016, 2017]
           if (sub["ANIO"] == y).any()]
    for y in yrs:
        sub[f"m_{y}"] = sub["mean_mmi"] * (sub["ANIO"] == y)
    b, se, t, p = wild_cluster_boot(sub, [f"m_{y}" for y in yrs], "m_2014")
    results.append({"estimate": "2014 investment pulse", "beta": b, "cluster_se": se,
                    "t": t, "p_wcb": p})
    print(f"2014 investment pulse: beta={b:.3f} se={se:.3f} WCB p={p:.4f}")

    # ---- 2) 2014 unskilled wage (REGNOCALD), event study focal = 2014 ----
    wcell = build_pooled_cell(lambda p: (pd.to_numeric(p.get("REGNOCALD"), errors="coerce"), "mean"))
    subw = detrend(wcell)
    yrsw = [y for y in [2003, 2004, 2005, 2006, 2007, 2008, 2010, 2011, 2013, 2014, 2015, 2016]
            if (subw["ANIO"] == y).any()]
    for y in yrsw:
        subw[f"m_{y}"] = subw["mean_mmi"] * (subw["ANIO"] == y)
    b, se, t, p = wild_cluster_boot(subw, [f"m_{y}" for y in yrsw], "m_2014")
    results.append({"estimate": "2014 unskilled wage", "beta": b, "cluster_se": se, "t": t, "p_wcb": p})
    print(f"2014 unskilled wage:   beta={b:.3f} se={se:.3f} WCB p={p:.4f}")

    # ---- 3) equipment average post-2010 deviation (net accumulation) ----
    eq = build_pooled_cell(lambda p: (np.where(p["ANIO"] <= 2014,
                                                _sum(p, ["CBNMAQ", "CBNVEH", "CBNSOF"]),
                                                _sum(p, ["CBMAQ", "CBVEH", "CBSOF"])), "sum"))
    sube = detrend(eq)
    sube["post_mmi"] = (sube["ANIO"] >= 2010) * sube["mean_mmi"]
    b, se, t, p = wild_cluster_boot(sube, ["post_mmi"], "post_mmi")
    results.append({"estimate": "equipment avg post (net)", "beta": b, "cluster_se": se, "t": t, "p_wcb": p})
    print(f"equipment avg post:    beta={b:.3f} se={se:.3f} WCB p={p:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "wild_bootstrap.csv", index=False)
    print(f"\nWrote {OUT/'wild_bootstrap.csv'}")


if __name__ == "__main__":
    main()
