*==============================================================================
* main_did.do -- Stata replication of the headline earthquake-DiD specifications
*
* Mirrors code/15_earthquake_did.py, 16_robustness.py and code/r/main_did.R.
* Reproduces, from the exported analysis cell panel, the four numbers the paper
* leans on (all verified identical in Python, R, and Stata):
*
*     (1) main DiD, investment, no trend               beta ~  0.517
*     (2) main DiD, investment, + region linear trend  beta ~  0.168
*     (3) raw event study, investment, 2014            beta ~  0.712
*     (4) pre-period-projected detrend, 2014 (HEADLINE) beta ~  0.579
*         + wild cluster bootstrap of the 2014 pulse   p    ~  0.03
*
* Spec:  log Y_{jrt} = beta (Post_t x MMI_r) + gamma_{jr} + delta_{jt} + e
*   j = 2-digit ISIC industry, r = region, t = year; MAIN sample j in {10,16,24}.
*   FE: industry x region (jr) and industry x year (jt); cluster: region.
*   Weights: effective establishment count (weight_sum), floored at 1.
*
* Run from the project root:  do code/stata/main_did.do
* Needs:  ssc install reghdfe ftools boottest
*
* Every estimate is compared with the Python value in data/python_benchmarks.csv
* (written by code/34_export_analysis_cell.py), and the script stops if one differs.
*==============================================================================

version 15
clear all
set more off

capture confirm file "data/analysis_cell.csv"
if _rc {
    di as err "Run from the project root after code/run_all.py: data/analysis_cell.csv not found"
    exit 601
}
capture mkdir "output/logs"
capture log close
log using "output/logs/stata_main_did.log", replace text

*--- Python benchmarks, stored as scalars py_<name> ------------------------------
import delimited "data/python_benchmarks.csv", varnames(1) clear
forvalues i = 1/`=_N' {
    scalar py_`=name[`i']' = value[`i']
}

* check_bm name value tol: stop if |value - py_name| >= tol
capture program drop check_bm
program define check_bm
    args name value tol
    local diff = `value' - scalar(py_`name')
    di as txt "  check `name': Stata " %9.6f `value' "  Python " %9.6f scalar(py_`name') "  diff " %9.2e `diff'
    if abs(`diff') >= `tol' {
        di as err "  `name' differs from the Python estimate"
        exit 9
    }
end

*--- load the exported analysis cell panel (written by the Python exporter) -----
import delimited "data/analysis_cell.csv", varnames(1) case(preserve) clear
capture destring ind2, replace force        // ind2 -> numeric if imported as string
gen double wclip = max(weight_sum, 1)        // weight floor (matches Python .clip(lower=1))

*==============================================================================
* (1)-(2)  MAIN DiD -- investment, no trend vs. + region linear trend
*==============================================================================
preserve
keep if inlist(ind2, 10, 16, 24)             // MAIN sample: Food, Wood, Basic metals
* fe_ok: drop missing outcome, then keep jr and jt cells with >=2 obs (sequential)
drop if missing(log_inv)
bysort jr_id: egen njr = count(log_inv)
keep if njr >= 2
bysort jt_id: egen njt = count(log_inv)
keep if njt >= 2

di as txt _n "--- (1) no trend  [expect +0.517] ---"
reghdfe log_inv post_mmi [aw=wclip], absorb(jr_id jt_id) vce(cluster REGION)
check_bm did_no_trend _b[post_mmi] 1e-4

di as txt _n "--- (2) + region linear trend  [expect +0.168] ---"
reghdfe log_inv post_mmi c.year_c#i.REGION [aw=wclip], absorb(jr_id jt_id) vce(cluster REGION)
check_bm did_linear_trend _b[post_mmi] 1e-4

*==============================================================================
* (3)  Raw event study -- investment x year, ref 2009 (2014 coefficient)
*==============================================================================
di as txt _n "--- (3) raw event study, 2014 x MMI  [expect +0.712] ---"
* Explicit year x MMI terms so that 2009 is the omitted reference year. The
* factor-variable form c.mean_mmi#ib2009.ANIO has no main effect to anchor the
* base, so it keeps 2009 and drops 2017 for collinearity instead.
foreach y in 2003 2004 2005 2006 2007 2008 2010 2011 2013 2014 2015 2016 2017 {
    gen double t_`y' = mean_mmi * (ANIO==`y')      // 2009 = reference (omitted)
}
reghdfe log_inv t_2003 t_2004 t_2005 t_2006 t_2007 t_2008 t_2010 t_2011 ///
                t_2013 t_2014 t_2015 t_2016 t_2017 [aw=wclip], ///
                absorb(jr_id jt_id) vce(cluster REGION)
di as res "  2014 pulse = " _b[t_2014]
check_bm es_raw_2014 _b[t_2014] 1e-4
restore

*==============================================================================
* (4)  Pre-period-projected detrend -- HEADLINE 2014 investment pulse
*      region linear trend estimated on PRE years (2003-2009) only, projected
*      forward; event study run on the detrended residual.
*==============================================================================
preserve
keep if inlist(ind2, 10, 16, 24)
drop if missing(log_inv)
bysort jr_id: egen njr = count(log_inv)
keep if njr >= 2
bysort jt_id: egen njt = count(log_inv)
keep if njt >= 2

gen byte pre = inrange(ANIO, 2003, 2009)     // PRE_YEARS = 2003..2009

* (1) industry x region pre-period weighted-mean level (raw weight_sum)
gen double _numpre = log_inv * weight_sum if pre
gen double _wpre   = weight_sum          if pre
bysort jr_id: egen double jr_num = total(_numpre)
bysort jr_id: egen double jr_den = total(_wpre)
gen double jr_mean = jr_num / jr_den
drop if missing(jr_mean)                     // jr must have pre-period obs
gen double dm = log_inv - jr_mean

* (2) per-region linear slope on the pre-period residual (guarded: >=3 obs, >=2 yrs)
gen double slope = 0
levelsof REGION, local(regs)
foreach r of local regs {
    quietly count if REGION==`r' & pre & !missing(dm)
    local nobs = r(N)
    quietly levelsof year_c if REGION==`r' & pre, local(yc)
    local nyc : word count `yc'
    if (`nobs' >= 3) & (`nyc' >= 2) {
        quietly regress dm year_c [aw=wclip] if REGION==`r' & pre
        quietly replace slope = _b[year_c] if REGION==`r'
    }
}

* (3) detrend all years; event study on the residual (jt FE only, jr level removed)
gen double detr = dm - slope * year_c
foreach y in 2003 2004 2005 2006 2007 2008 2010 2011 2013 2014 2015 2016 2017 {
    gen double t_`y' = mean_mmi * (ANIO==`y')      // 2009 = reference (omitted)
}

di as txt _n "--- (4) pre-projected detrend, 2014 & 2015 pulse  [expect +0.579 / +0.602] ---"
reghdfe detr t_2003 t_2004 t_2005 t_2006 t_2007 t_2008 t_2010 t_2011 ///
             t_2013 t_2014 t_2015 t_2016 t_2017 [aw=wclip], ///
             absorb(jt_id) vce(cluster REGION)
di as res "  2014 pulse = " _b[t_2014] "   2015 pulse = " _b[t_2015]
check_bm es_pretrend_2014 _b[t_2014] 1e-4
check_bm es_pretrend_2015 _b[t_2015] 1e-4

* wild cluster bootstrap of the 2014 pulse (restricted, Rademacher, B=999)
di as txt _n "--- wild cluster bootstrap of the 2014 pulse  [expect p ~ 0.03] ---"
boottest t_2014, reps(999) weighttype(rademacher) seed(20100227) nograph
* The bootstrap p-value depends on the random draws, so it only has to be close.
check_bm wcb_p_2014 r(p) 0.02
restore

di as txt _n "OK: Stata reproduces the Python estimates."
log close
