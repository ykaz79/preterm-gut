"""Supplementary Table S8: sensitivity of the pre-onset comparison to distributional assumptions.

Six specifications comparing pre-onset samples with samples of infants without
complications, for Klebsiella and summed Enterobacteriaceae, in the whole cohort and
within Site B.  Infant is the random intercept or clustering unit in every model except
the beta regression.  Also reports the Shapiro-Wilk test and skewness of the residuals
of the untransformed mixed models (Supplementary Methods).

Requires d_final.pkl written by batch_models.py (samples at or after onset excluded).
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.othermod.betareg import BetaModel
from scipy import stats

warnings.filterwarnings("ignore")
d = pd.read_pickle("d_final.pkl")
SETS = {"whole cohort": d, "Site B": d[d.site == "B"]}
TAXA = {"Klebsiella": "Klebsiella", "Summed Enterobacteriaceae": "Summed_Entero"}
LOGIT_C = 0.005      # continuity constant for the logit transformation (proportion scale)
LOG_C = 0.01         # constant added before log10 (percentage points)
DETECT = 0.01        # detection threshold (% relative abundance)
N_BOOT, SEED = 4000, 20260726


def fmt(est, lo, hi, p, d=2):
    ps = "P < 0.0001" if round(p, 4) == 0 else f"P = {p:.4f}"
    return f"{est:.{d}f} ({lo:.{d}f} to {hi:.{d}f}), {ps}"


def lmm(df, y):
    m = smf.mixedlm(f"{y} ~ y", df, groups=df["case"]).fit(reml=False)
    lo, hi = m.conf_int().loc["y"]
    return m, (m.params["y"], lo, hi, m.pvalues["y"])


def hodges_lehmann(a, b):
    return np.median(np.subtract.outer(a, b))


def column(df, var):
    df = df.copy()
    p = df[var] / 100
    out = {}
    m, r = lmm(df, var)
    out["Linear mixed model, untransformed"] = fmt(*r, d=1)
    resid = {"shapiro_p": stats.shapiro(m.resid).pvalue, "skew": stats.skew(m.resid)}
    df["logit_"] = np.log((p + LOGIT_C) / (1 - p + LOGIT_C))
    out["Linear mixed model, logit transformed"] = fmt(*lmm(df, "logit_")[1])
    df["log10_"] = np.log10(df[var] + LOG_C)
    out["Linear mixed model, log10 transformed"] = fmt(*lmm(df, "log10_")[1])
    n = len(df)
    yb = (p * (n - 1) + 0.5) / n                         # Smithson-Verkuilen compression
    bm = BetaModel(yb, sm.add_constant(df["y"].astype(float))).fit(disp=0)
    lo, hi = bm.conf_int().loc["y"]
    out["Beta regression"] = fmt(bm.params["y"], lo, hi, bm.pvalues["y"])
    df["detected"] = (df[var] >= DETECT).astype(int)
    ge = smf.gee("detected ~ y", groups=df["case"], data=df, family=sm.families.Binomial(),
                 cov_struct=sm.cov_struct.Exchangeable()).fit()
    lo, hi = np.exp(ge.conf_int().loc["y"])
    out["Two-part model, part 1: odds of detection"] = fmt(np.exp(ge.params["y"]), lo, hi, ge.pvalues["y"])
    det = df[df.detected == 1]
    out["Two-part model, part 2: abundance where detected"] = fmt(*lmm(det, "log10_")[1]) + f" [n = {len(det)}]"
    med = df.groupby("case").agg(m=(var, "median"), y=("y", "max"))
    a = med.m[med.y == 1].values; b = med.m[med.y == 0].values
    rng = np.random.default_rng(SEED)
    boot = [hodges_lehmann(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))])
            for _ in range(N_BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    pmw = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    out["Rank-based, infant-level medians (Hodges-Lehmann)"] = fmt(hodges_lehmann(a, b), lo, hi, pmw, d=1)
    return out, resid


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    cols, res = {}, []
    for tname, var in TAXA.items():
        for sname, df in SETS.items():
            c, r = column(df, var)
            cols[f"{tname}, {sname}"] = c
            res.append(dict(model=f"{tname}, {sname}", shapiro_wilk_P=f"{r['shapiro_p']:.1e}", skewness=round(r["skew"], 2)))
    print(pd.DataFrame(cols).to_string())
    print("\nResiduals of the untransformed linear mixed models")
    print(pd.DataFrame(res).to_string(index=False))
