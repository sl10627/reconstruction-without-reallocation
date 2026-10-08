# =============================================================================
# main_did.R  --  R replication of the headline earthquake-DiD specifications
#
# Mirrors the Python pipeline (code/15_earthquake_did.py, 16_robustness.py) for
# the 2010 Maule earthquake capital-shock DiD. Reproduces, from the exported
# analysis cell panel, the four numbers the paper leans on:
#
#     (1) main DiD, investment, no trend            beta ~  0.517
#     (2) main DiD, investment, +region trend       beta ~  0.168
#     (3) raw event study, investment, 2014          beta ~  0.712
#     (4) pre-period-projected detrend, 2014 (HEADLINE) beta ~ 0.579
#
# Specification:  log Y_{jrt} = beta (Post_t x MMI_r) + gamma_{jr} + delta_{jt}
#   j = 2-digit ISIC industry, r = region, t = year; MAIN sample j in {10,16,24}.
#   FE: industry x region (jr) and industry x year (jt). Cluster: region.
#   Weights: effective establishment count (weight_sum), floored at 1 (matches
#   the Python .clip(lower=1)).
#
# Run:  Rscript code/r/main_did.R
# Needs: fixest  (install.packages("fixest")). Base read.csv -- no arrow needed.
# =============================================================================

suppressMessages(library(fixest))

# locate project root: from the --file= script path if present, else the cwd
.args <- commandArgs(FALSE)
.sf <- sub("^--file=", "", .args[grepl("^--file=", .args)])
ROOT <- if (length(.sf) == 1) normalizePath(file.path(dirname(.sf), "..", "..")) else getwd()
if (!dir.exists(file.path(ROOT, "data"))) ROOT <- getwd()

d <- read.csv(file.path(ROOT, "data", "analysis_cell.csv"), stringsAsFactors = FALSE)
d$wclip <- pmax(d$weight_sum, 1)          # weight floor, matches Python .clip(lower=1)
MAIN <- c(10, 16, 24)

# ---- fe_ok: drop missing outcome, then keep jr and jt cells with >=2 obs ----
# (sequential, exactly as the Python helper: filter jr first, then jt)
fe_ok <- function(df, ycol) {
  df <- df[!is.na(df[[ycol]]), ]
  for (k in c("jr", "jt")) {
    tab <- table(df[[k]])
    df <- df[df[[k]] %in% names(tab)[tab >= 2], ]
  }
  df
}

cat("=======================================================================\n")
cat(" (1)-(2)  MAIN DiD -- investment, no trend vs. + region linear trend\n")
cat("=======================================================================\n")
sub <- fe_ok(d[d$ind2 %in% MAIN, ], "log_inv")

m_none <- feols(log_inv ~ post_mmi | jr + jt,
                data = sub, weights = ~wclip, cluster = ~REGION)
# region-specific linear trend = factor(REGION) interacted with continuous year_c
m_lin  <- feols(log_inv ~ post_mmi + i(REGION, year_c) | jr + jt,
                data = sub, weights = ~wclip, cluster = ~REGION)

cat(sprintf("  no trend      : beta = %+.4f  (se %.4f, p %.4f, n %d)\n",
            coef(m_none)["post_mmi"], se(m_none)["post_mmi"],
            pvalue(m_none)["post_mmi"], m_none$nobs))
cat(sprintf("  +region trend : beta = %+.4f  (se %.4f, p %.4f, n %d)\n",
            coef(m_lin)["post_mmi"], se(m_lin)["post_mmi"],
            pvalue(m_lin)["post_mmi"], m_lin$nobs))

cat("\n=======================================================================\n")
cat(" (3)  Raw event study -- investment x year, ref 2009 (2014 coefficient)\n")
cat("=======================================================================\n")
m_es <- feols(log_inv ~ i(ANIO, mean_mmi, ref = 2009) | jr + jt,
              data = sub, weights = ~wclip, cluster = ~REGION)
b2014_raw <- coef(m_es)["ANIO::2014:mean_mmi"]
cat(sprintf("  2014 x MMI (raw) : beta = %+.4f  (se %.4f)\n",
            b2014_raw, se(m_es)["ANIO::2014:mean_mmi"]))

cat("\n=======================================================================\n")
cat(" (4)  Pre-period-projected detrend -- HEADLINE 2014 investment pulse\n")
cat("     region linear trend estimated on PRE years (2003-2009) only, then\n")
cat("     projected forward; event study run on the detrended residual.\n")
cat("=======================================================================\n")
PRE <- c(2003, 2004, 2005, 2006, 2007, 2008, 2009)
sp  <- fe_ok(d[d$ind2 %in% MAIN, ], "log_inv")
pre <- sp[sp$ANIO %in% PRE, ]

# (1) industry x region pre-period weighted mean level (raw weight_sum, matches Python)
jr_mean <- tapply(seq_len(nrow(pre)), pre$jr, function(ix)
  sum(pre$log_inv[ix] * pre$weight_sum[ix]) / sum(pre$weight_sum[ix]))
sp <- sp[sp$jr %in% names(jr_mean), ]
sp$dm <- sp$log_inv - jr_mean[sp$jr]

# (2) per-region linear slope on the pre-period residual (weights floored)
slopes <- sapply(sort(unique(sp$REGION)), function(r) {
  dd <- sp[sp$REGION == r & sp$ANIO %in% PRE, ]
  if (length(unique(dd$year_c)) >= 2 && nrow(dd) >= 3) {
    coef(lm(dm ~ year_c, data = dd, weights = pmax(dd$weight_sum, 1)))["year_c"]
  } else 0
})
names(slopes) <- sort(unique(sp$REGION))

# (3) detrend all years, event study on the residual (jt FE only; jr level removed)
sp$detr <- as.numeric(sp$dm) - as.numeric(slopes[as.character(sp$REGION)]) * sp$year_c
m_pt <- feols(detr ~ i(ANIO, mean_mmi, ref = 2009) | jt,
              data = sp, weights = ~wclip, cluster = ~REGION)
b2014 <- coef(m_pt)["ANIO::2014:mean_mmi"]
b2015 <- coef(m_pt)["ANIO::2015:mean_mmi"]
cat(sprintf("  2014 x MMI (pre-projected) : beta = %+.4f  (se %.4f, p %.4f)\n",
            b2014, se(m_pt)["ANIO::2014:mean_mmi"], pvalue(m_pt)["ANIO::2014:mean_mmi"]))
cat(sprintf("  2015 x MMI (pre-projected) : beta = %+.4f  (se %.4f, p %.4f)\n",
            b2015, se(m_pt)["ANIO::2015:mean_mmi"], pvalue(m_pt)["ANIO::2015:mean_mmi"]))

# ---- wild cluster bootstrap of the 2014 pulse (restricted, Rademacher) -------
# fwildclusterboot is not available for R >= 4.6, so the WCB-t is hand-rolled to
# match code/22_wild_bootstrap.py: null-imposed (restricted) residuals, Rademacher
# cluster weights, CR1 small-sample adjustment, p = (1 + #{|t*|>=|t|}) / (B + 1).
# In Stata this one line is boottest (see code/stata/main_did.do).
wls_cr <- function(X, y, w, groups, focal) {          # beta_focal, cluster-robust t
  sw <- sqrt(w); Xw <- X * sw; yw <- y * sw
  XtXi <- solve(crossprod(Xw)); beta <- XtXi %*% crossprod(Xw, yw)
  u <- as.numeric(y - X %*% beta)
  meat <- matrix(0, ncol(X), ncol(X))
  for (g in unique(groups)) { ix <- groups == g
    sg <- crossprod(X[ix, , drop = FALSE] * w[ix], u[ix]); meat <- meat + tcrossprod(sg) }
  G <- length(unique(groups)); n <- nrow(X); k <- ncol(X)
  V <- XtXi %*% meat %*% XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
  list(t = beta[focal] / sqrt(V[focal, focal]), beta = beta)
}
ES <- c(2003,2004,2005,2006,2007,2008,2010,2011,2013,2014,2015,2016,2017)  # 2009 = ref
tmat <- sapply(ES, function(y) sp$mean_mmi * (sp$ANIO == y)); colnames(tmat) <- paste0("t_", ES)
X <- cbind(1, tmat, model.matrix(~ factor(sp$jt))[, -1])
y <- as.numeric(sp$detr); w <- sp$wclip; grp <- sp$REGION
focal <- which(colnames(tmat) == "t_2014") + 1L          # +1 for the leading intercept
obs <- wls_cr(X, y, w, grp, focal); t_hat <- obs$t
# restricted fit: impose the 2014 coefficient = 0 (drop that column), get residuals
keep <- setdiff(seq_len(ncol(X)), focal); sw <- sqrt(w)
br <- qr.solve(X[, keep] * sw, y * sw); yhat_r <- as.numeric(X[, keep] %*% br); ur <- y - yhat_r
set.seed(20100227); B <- 999; uniq <- unique(grp); cnt <- 0
for (b in seq_len(B)) {
  wg <- setNames(sample(c(-1, 1), length(uniq), replace = TRUE), uniq)
  ystar <- yhat_r + wg[as.character(grp)] * ur
  if (abs(wls_cr(X, ystar, w, grp, focal)$t) >= abs(t_hat)) cnt <- cnt + 1
}
cat(sprintf("\n  wild cluster bootstrap (B=999, Rademacher, restricted): t=%.2f, p = %.3f\n",
            t_hat, (1 + cnt) / (B + 1)))

p_wcb <- (1 + cnt) / (B + 1)

# ---- compare with the Python estimates (data/python_benchmarks.csv, written by
# code/34_export_analysis_cell.py). Point estimates and the bootstrap t must agree
# to 1e-4; the bootstrap p-value depends on the random draws, so it only has to be
# within 0.02 of Python's.
bm <- read.csv(file.path(ROOT, "data", "python_benchmarks.csv"))
py <- setNames(bm$value, bm$name)
r_est <- c(did_no_trend = unname(coef(m_none)["post_mmi"]),
           did_linear_trend = unname(coef(m_lin)["post_mmi"]),
           es_raw_2014 = unname(b2014_raw),
           es_pretrend_2014 = unname(b2014),
           es_pretrend_2015 = unname(b2015),
           wcb_t_2014 = unname(t_hat))
cmp <- data.frame(estimate = names(r_est), R = round(r_est, 4),
                  Python = round(py[names(r_est)], 4), diff = signif(r_est - py[names(r_est)], 2))
cat("\nR vs Python:\n"); print(cmp, row.names = FALSE)
cat(sprintf("  wild bootstrap p: R %.3f, Python %.3f\n", p_wcb, py["wcb_p_2014"]))
stopifnot(all(abs(r_est - py[names(r_est)]) < 1e-4),
          abs(p_wcb - py["wcb_p_2014"]) < 0.02)
cat("OK: R reproduces the Python estimates.\n")
