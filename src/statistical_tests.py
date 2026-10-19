"""
src/statistical_tests.py
========================
The complete test battery the README promised: descriptive profile, correlation,
chi-square, ANOVA, normality diagnostics and non-parametric alternatives.

The previous README claimed "normality and non-parametric tests" while the notebook only
contained chi-square and ANOVA, so the claim was not reproducible. This module holds every
test, the notebook calls it, and src/reports.py writes the same tables to the Excel report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from data_pipeline import CLEAN_PATH, build_all

ALPHA = 0.05


def _verdict(p: float, reject_msg: str, keep_msg: str) -> str:
    return reject_msg if p < ALPHA else keep_msg


# ------------------------------------------------------------------ data
def load_daily() -> pd.DataFrame:
    """Daily series from the pipeline (gap days kept as zero, which is documented)."""
    out = build_all(verbose=False)
    return out["daily"]


def load_records() -> pd.DataFrame:
    return build_all(verbose=False)["df"]


# ------------------------------------------------------- descriptive profile
def descriptive_profile(daily: pd.DataFrame) -> pd.DataFrame:
    cols = ["Cases", "Deaths", "Recovered", "Active", "Cumulative_Cases"]
    cols = [c for c in cols if c in daily.columns]
    rows = []
    for c in cols:
        s = daily[c].dropna()
        rows.append({
            "variable": c,
            "n": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "cv_%": (s.std() / s.mean() * 100) if s.mean() else np.nan,
            "min": s.min(),
            "max": s.max(),
            "skewness": s.skew(),
            "kurtosis": s.kurtosis(),
            "q1": s.quantile(.25),
            "q3": s.quantile(.75),
        })
    return pd.DataFrame(rows).round(4)


# -------------------------------------------------------------- correlation
def correlation_analysis(daily: pd.DataFrame) -> dict:
    cols = [c for c in ["Cases", "Deaths", "Recovered", "Active", "Cumulative_Cases"]
            if c in daily.columns]
    sub = daily[cols].dropna()
    matrix = sub.corr()

    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            r, p = stats.pearsonr(sub[a], sub[b])
            rho, p_s = stats.spearmanr(sub[a], sub[b])
            rows.append({
                "pair": f"{a} vs {b}",
                "pearson_r": r, "pearson_p": p,
                "spearman_rho": rho, "spearman_p": p_s,
                "significant": "yes" if p < ALPHA else "no",
            })
    return dict(matrix=matrix, pairs=pd.DataFrame(rows).round(6))


# ------------------------------------------------------- chi-square / ANOVA
def chi_square_province_travel(df: pd.DataFrame) -> dict:
    tab = pd.crosstab(df["Province"], df["Travel_history"])
    chi2, p, dof, expected = stats.chi2_contingency(tab)
    n = tab.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
    return dict(
        table=tab, chi2=float(chi2), p=float(p), dof=int(dof),
        cramers_v=float(cramers_v),
        verdict=_verdict(p, "province and exposure source are dependent",
                         "province and exposure source are independent"),
    )


def anova_cases_by_province(df: pd.DataFrame) -> dict:
    groups = [g["Cases"].values for _, g in df.groupby("Province") if len(g) > 1]
    names = [p for p, g in df.groupby("Province") if len(g) > 1]
    f, p = stats.f_oneway(*groups)

    grand = df["Cases"].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((df["Cases"] - grand) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total else np.nan

    tukey = None
    try:
        tukey = pairwise_tukeyhsd(df["Cases"].values, df["Province"].values, alpha=ALPHA)
    except Exception:
        pass

    summary = pd.DataFrame({
        "Province": names,
        "n": [len(g) for g in groups],
        "mean": [g.mean() for g in groups],
        "std": [g.std() for g in groups],
    }).sort_values("mean", ascending=False).round(3)

    return dict(f=float(f), p=float(p), eta_squared=float(eta_sq),
                summary=summary, tukey=tukey,
                verdict=_verdict(p, "at least one province differs",
                                 "no evidence that province means differ"))


# ------------------------------------------------------------------ normality
def normality_tests(daily: pd.DataFrame, column: str = "Cases") -> dict:
    s = daily[column].dropna()
    sw_stat, sw_p = stats.shapiro(s)
    dp_stat, dp_p = stats.normaltest(s)                 # D'Agostino-Pearson
    jb_stat, jb_p = stats.jarque_bera(s)
    ks_stat, ks_p = stats.kstest((s - s.mean()) / s.std(), "norm")

    return dict(
        variable=column, n=len(s),
        shapiro={"stat": float(sw_stat), "p": float(sw_p)},
        dagostino={"stat": float(dp_stat), "p": float(dp_p)},
        jarque_bera={"stat": float(jb_stat), "p": float(jb_p)},
        kolmogorov_smirnov={"stat": float(ks_stat), "p": float(ks_p)},
        skew=float(s.skew()), kurtosis=float(s.kurtosis()),
        verdict=_verdict(sw_p, "not normal - prefer non-parametric tests",
                         "cannot reject normality"),
        table=pd.DataFrame([
            {"test": "Shapiro-Wilk", "statistic": float(sw_stat), "p_value": float(sw_p),
             "null": "data are normal",
             "verdict": "reject H0" if sw_p < ALPHA else "fail to reject H0"},
            {"test": "D'Agostino-Pearson", "statistic": float(dp_stat), "p_value": float(dp_p),
             "null": "data are normal",
             "verdict": "reject H0" if dp_p < ALPHA else "fail to reject H0"},
            {"test": "Jarque-Bera", "statistic": float(jb_stat), "p_value": float(jb_p),
             "null": "data are normal",
             "verdict": "reject H0" if jb_p < ALPHA else "fail to reject H0"},
            {"test": "Kolmogorov-Smirnov", "statistic": float(ks_stat), "p_value": float(ks_p),
             "null": "data are normal",
             "verdict": "reject H0" if ks_p < ALPHA else "fail to reject H0"},
        ]),
    )


# ------------------------------------------------------- non-parametric tests
def nonparametric_tests(df: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Because the daily series is not normal, mirror the parametric tests with rank tests."""
    out = {}

    # Mann-Whitney U: local transmission vs imported/unknown records
    local = df.loc[df["Is_Local"] == 1, "Cases"].values
    imported = df.loc[df["Is_Local"] == 0, "Cases"].values
    if len(local) and len(imported):
        u, p = stats.mannwhitneyu(local, imported, alternative="two-sided")
        out["mann_whitney_local_vs_imported"] = {
            "u": float(u), "p": float(p),
            "median_local": float(np.median(local)),
            "median_imported": float(np.median(imported)),
            "verdict": _verdict(p, "record size differs by transmission type",
                                "no evidence that record size differs"),
        }

    # Kruskal-Wallis: cases across provinces (the non-parametric ANOVA)
    groups = [g["Cases"].values for _, g in df.groupby("Province") if len(g) > 1]
    if len(groups) > 2:
        h, p = stats.kruskal(*groups)
        out["kruskal_wallis_province"] = {
            "h": float(h), "p": float(p),
            "verdict": _verdict(p, "province distributions differ",
                                "no evidence that province distributions differ"),
        }

    # Wilcoxon signed-rank: daily cases vs daily deaths (paired, same day)
    a, b = daily["Cases"].values, daily["Deaths"].values
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() > 10:
        w, p = stats.wilcoxon(a[mask], b[mask])
        out["wilcoxon_cases_vs_deaths"] = {
            "w": float(w), "p": float(p),
            "verdict": _verdict(p, "cases and deaths differ in level",
                                "no evidence that the levels differ"),
        }

    # Spearman on the daily series (rank correlation, robust to the skew)
    rho, p = stats.spearmanr(daily["Cases"], daily["Deaths"])
    out["spearman_cases_deaths"] = {
        "rho": float(rho), "p": float(p),
        "verdict": _verdict(p, "monotone association present",
                            "no monotone association detected"),
    }
    return out


# --------------------------------------------------------------- run all
def run_all(verbose: bool = True) -> dict:
    daily = load_daily()
    df = load_records()

    res = {
        "descriptive": descriptive_profile(daily),
        "correlation": correlation_analysis(daily),
        "chi_square": chi_square_province_travel(df),
        "anova": anova_cases_by_province(df),
        "normality": normality_tests(daily),
        "nonparametric": nonparametric_tests(df, daily),
        "n_records": len(df),
        "n_days": len(daily),
    }

    if verbose:
        print("=" * 78)
        print("DESCRIPTIVE PROFILE")
        print("=" * 78)
        print(res["descriptive"].to_string(index=False))
        print()
        print("=" * 78)
        print("CHI-SQUARE: province vs exposure source")
        print("=" * 78)
        c = res["chi_square"]
        print("  chi2 = %.3f  dof = %d  p = %.4e  Cramer's V = %.4f"
              % (c["chi2"], c["dof"], c["p"], c["cramers_v"]))
        print("  ->", c["verdict"])
        print()
        print("=" * 78)
        print("ONE-WAY ANOVA: record cases across provinces")
        print("=" * 78)
        a = res["anova"]
        print(a["summary"].to_string(index=False))
        print("\n  F = %.3f  p = %.4e  eta squared = %.4f" % (a["f"], a["p"], a["eta_squared"]))
        print("  ->", a["verdict"])
        print()
        print("=" * 78)
        print("NORMALITY of daily cases")
        print("=" * 78)
        print(res["normality"]["table"].to_string(index=False))
        print("  ->", res["normality"]["verdict"])
        print()
        print("=" * 78)
        print("NON-PARAMETRIC TESTS")
        print("=" * 78)
        for name, v in res["nonparametric"].items():
            stat = [k for k in v if k not in ("verdict", "median_local", "median_imported")]
            print("  %-34s %s  p=%.4e  -> %s"
                  % (name, {k: round(v[k], 4) for k in stat}, v["p"], v["verdict"]))
    return res


if __name__ == "__main__":
    run_all()
