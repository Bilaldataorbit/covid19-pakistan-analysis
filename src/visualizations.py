"""
src/visualizations.py
=====================
Every static chart and every interactive chart the portfolio ships, generated from the
cleaned pipeline. Figures are written to outputs/figures and outputs/interactive.

A note on the interactive HTML files: they are saved with
`include_plotlyjs="cdn"` so each file carries only the chart data (about 5-15 KB) instead
of a 4.6 MB copy of the Plotly library. The previous build embedded the library in all six
files, which is why the repository was 48 MB.
"""
from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from data_pipeline import ROOT, build_all
from ml_models import (build_candidates, feature_importance, load_model_table,
                       time_split, train_and_evaluate)

warnings.filterwarnings("ignore")

FIG_DIR = os.path.join(ROOT, "outputs", "figures")
INT_DIR = os.path.join(ROOT, "outputs", "interactive")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(INT_DIR, exist_ok=True)

PALETTE = "viridis"
MAIN, ACC, WARN = "#2E75B6", "#BF8F00", "#C00000"
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.autolayout": False,
})

_saved: list[str] = []


def save(fig, name: str, tight=True):
    path = os.path.join(FIG_DIR, name)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _saved.append(name)
    print("  saved", name)


def numbered(i: int, slug: str) -> str:
    return f"{i:02d}_{slug}.png"


# ----------------------------------------------------------------- data
def load():
    return build_all(verbose=False)


# ------------------------------------------------------------- figures
def fig_overview(out):
    df = out["df"]
    daily = out["daily"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    ax = axes[0, 0]
    ax.plot(daily["Date"], daily["Cumulative_Cases"], color=MAIN, lw=2.2, label="Cases")
    ax.plot(daily["Date"], daily["Cumulative_Deaths"], color=WARN, lw=2, label="Deaths")
    ax.plot(daily["Date"], daily["Cumulative_Recovered"], color="#375623", lw=2, label="Recovered")
    ax.set_title("Cumulative trajectory")
    ax.set_ylabel("people")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.tick_params(axis="x", rotation=30)

    ax = axes[0, 1]
    prov = out["province"]
    ax.barh(prov["Province"][::-1], prov["Cases"][::-1], color=sns.color_palette(PALETTE, len(prov)))
    ax.set_title("Total cases by province")
    ax.set_xlabel("cases")
    for y, (v, s) in enumerate(zip(prov["Cases"][::-1], prov["Case_Share"][::-1])):
        ax.text(v, y, f"  {s:.1f}%", va="center", fontsize=8)

    ax = axes[1, 0]
    ax.plot(daily["Date"], daily["Cases"], color="#9aa5b1", lw=1, alpha=0.8, label="daily")
    ax.plot(daily["Date"], daily["MA7_Cases"], color=MAIN, lw=2.4, label="7-day average")
    ax.set_title("Daily new cases with 7-day moving average")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 1]
    city = out["city"].head(10)
    ax.barh(city["City"][::-1], city["Cases"][::-1], color=sns.color_palette("magma", len(city)))
    ax.set_title("Top 10 cities by cases")
    ax.set_xlabel("cases")
    fig.suptitle("COVID-19 Pakistan - overview", fontsize=15, fontweight="bold")
    save(fig, numbered(1, "overview_dashboard"))


def fig_daily_trends(daily):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax = axes[0]
    ax.fill_between(daily["Date"], daily["Cases"], color=MAIN, alpha=0.35)
    ax.plot(daily["Date"], daily["Cases"], color=MAIN, lw=1.6)
    ax.plot(daily["Date"], daily["MA7_Cases"], color=WARN, lw=2.2, label="7-day mean")
    ax.set_title("Daily cases")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.fill_between(daily["Date"], daily["Deaths"], color=WARN, alpha=0.35)
    ax.plot(daily["Date"], daily["Deaths"], color=WARN, lw=1.6)
    ax.plot(daily["Date"], daily["MA7_Deaths"], color="#5a3a00", lw=2.2, label="7-day mean")
    ax.set_title("Daily deaths")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.tick_params(axis="x", rotation=30)
    save(fig, numbered(2, "daily_trends"))


def fig_cumulative_rates(daily):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    ax.plot(daily["Date"], daily["CFR"], color=WARN, lw=2, label="CFR %")
    ax.plot(daily["Date"], daily["Recovery_Rate"], color="#375623", lw=2, label="Recovery %")
    ax.set_title("Cumulative case fatality and recovery rate")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1]
    ax.plot(daily["Date"], daily["Growth_Rate"], color=MAIN, lw=1.2, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Day-over-day growth rate (%)")
    ax.set_ylabel("%")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.tick_params(axis="x", rotation=30)
    save(fig, numbered(3, "rates_and_growth"))


def fig_province_city(out):
    prov, city, df = out["province"], out["city"], out["df"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    order = prov.sort_values("Cases")["Province"]
    sns.boxplot(data=df, y="Province", x="Cases", order=order,
                ax=ax, palette=PALETTE, hue="Province", legend=False, showfliers=False)
    ax.set_title("Daily cases distribution by province")
    ax.set_xlabel("cases in a single record")

    ax = axes[1]
    top = city.head(12)
    ax.bar(range(len(top)), top["Cases"], color=sns.color_palette("viridis", len(top)))
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["City"], rotation=45, ha="right", fontsize=8)
    ax.set_title("Top 12 cities")
    ax.set_ylabel("cases")
    for i, v in enumerate(top["Cases"]):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7)
    save(fig, numbered(4, "province_city_distribution"))


def fig_province_cfr(prov):
    fig, ax = plt.subplots(figsize=(12, 6))
    p = prov.sort_values("CFR")
    colors = [WARN if v > p["CFR"].median() else MAIN for v in p["CFR"]]
    bars = ax.barh(p["Province"], p["CFR"], color=colors)
    ax.axvline(prov["Deaths"].sum() / prov["Cases"].sum() * 100, color="black",
               ls="--", lw=1.2, label="national CFR")
    ax.set_title("Case fatality rate by province")
    ax.set_xlabel("CFR %")
    ax.legend(frameon=False)
    for b, v in zip(bars, p["CFR"]):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.2f}%", va="center", fontsize=8)
    save(fig, numbered(5, "province_cfr"))


def fig_monthly_heatmap(out):
    df = out["df"]
    piv = df.pivot_table(values="Cases", index="Province", columns="Month_Short",
                         aggfunc="sum", fill_value=0)
    months = [m for m in ["Feb", "Mar", "Apr", "May", "Jun"] if m in piv.columns]
    piv = piv[months]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(piv, annot=True, fmt=",.0f", cmap="YlOrRd", ax=ax, linewidths=.5)
    ax.set_title("Cases by province and month")
    ax.set_ylabel("")
    save(fig, numbered(6, "heatmap_province_month"))


def fig_travel(out):
    t = out["travel"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    top = t.head(8)
    ax.pie(top["Cases"], labels=top["Travel_history"], autopct="%1.1f%%",
           colors=sns.color_palette(PALETTE, len(top)), startangle=90)
    ax.set_title("Case share by exposure source (top 8)")

    ax = axes[1]
    ax.barh(t["Travel_history"][::-1], t["Cases"][::-1],
            color=sns.color_palette("viridis", len(t)))
    ax.set_xscale("symlog")
    ax.set_title("Cases by exposure source (log scale)")
    ax.set_xlabel("cases (symlog)")
    save(fig, numbered(7, "travel_history"))


def fig_distributions(out):
    daily, df = out["daily"], out["df"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    sns.histplot(daily["Cases"], bins=25, kde=True, ax=ax, color=MAIN)
    ax.axvline(daily["Cases"].mean(), color=WARN, ls="--", label="mean")
    ax.axvline(daily["Cases"].median(), color="#375623", ls=":", label="median")
    ax.set_title("Distribution of daily cases")
    ax.legend(frameon=False)

    ax = axes[1]
    sns.histplot(df["Cases"], bins=40, ax=ax, color="#6c8ebf", log_scale=(False, True))
    ax.set_title("Record-level cases (log count)")
    ax.set_xlabel("cases in a record")

    ax = axes[2]
    sns.histplot(daily["Growth_Rate"].dropna(), bins=30, kde=True, ax=ax, color=WARN)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Daily growth rate")
    ax.set_xlabel("%")
    save(fig, numbered(8, "distributions"))


def fig_correlation(daily, df):
    cols = ["Cases", "Deaths", "Recovered", "Active", "Cumulative_Cases",
            "MA7_Cases", "CFR", "Recovery_Rate"]
    sub = daily[cols].dropna()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sub.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax, cbar_kws={"shrink": .8})
    ax.set_title("Correlation matrix - daily series")
    save(fig, numbered(9, "correlation_matrix"))

    num = df[["Cases", "Deaths", "Recovered", "CFR", "Recovery_Rate"]].dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax)
    ax.set_title("Correlation matrix - record level")
    save(fig, numbered(10, "correlation_records"))


def fig_scatter(daily, df):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    ax.scatter(daily["Cases"], daily["Deaths"], s=28, alpha=.75, color=WARN)
    ax.set_xlabel("daily cases");
    ax.set_ylabel("daily deaths")
    ax.set_title("Cases vs deaths")

    ax = axes[1]
    ax.scatter(daily["Cumulative_Cases"], daily["Cumulative_Recovered"],
               s=28, alpha=.75, color="#375623")
    ax.set_xlabel("cumulative cases");
    ax.set_ylabel("cumulative recovered")
    ax.set_title("Cases vs recoveries (cumulative)")

    ax = axes[2]
    prov = df.groupby("Province").agg(Cases=("Cases", "sum"), CFR=("CFR", "mean")).reset_index()
    ax.scatter(prov["Cases"], prov["CFR"], s=140, color=MAIN, alpha=.8)
    for _, r in prov.iterrows():
        ax.annotate(r["Province"][:14], (r["Cases"], r["CFR"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("total cases (log)")
    ax.set_ylabel("mean CFR %")
    ax.set_title("Province size vs fatality")
    save(fig, numbered(11, "scatter_relationships"))


def fig_ml(res, out):
    test = res["test"]
    y = test["Cases"].values
    pred = res["y_pred"]
    dates = pd.to_datetime(test["Date"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    ax.plot(dates, y, "o-", color=MAIN, label="actual", ms=4)
    ax.plot(dates, pred, "s--", color=WARN, label="predicted", ms=4)
    ax.plot(dates, test["Lag_Cases_1"], ":", color="#375623", label="naive (t-1)")
    ax.set_title(f"Hold-out forecast - best learner: {res['best_name']}")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=30)

    ax = axes[0, 1]
    ax.scatter(y, pred, s=45, color=WARN, alpha=.85, label="learner")
    ax.scatter(y, test["Lag_Cases_1"], s=45, color="#375623", alpha=.55, label="naive")
    lim = [0, max(y.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, color="black", lw=1, ls="--")
    ax.set_xlabel("actual");
    ax.set_ylabel("predicted")
    ax.set_title("Actual vs predicted (45 degree line = perfect)")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    names = [k for k in res["results"] if k != "Naive (tomorrow = today)"]
    rmse = [res["results"][k]["RMSE"] for k in names]
    naive = res["results"]["Naive (tomorrow = today)"]["RMSE"]
    order = np.argsort(rmse)
    bars = ax.barh([names[i] for i in order], [rmse[i] for i in order], color=MAIN)
    ax.axvline(naive, color=WARN, ls="--", lw=1.6, label=f"naive RMSE {naive:.0f}")
    ax.set_title("Hold-out RMSE by model (lower is better)")
    ax.legend(frameon=False)
    for b in bars:
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {b.get_width():.0f}",
                va="center", fontsize=8)

    ax = axes[1, 1]
    imp = feature_importance(res["best_model"], res["features"]).head(10)
    ax.barh(imp["Feature"][::-1], imp["Importance"][::-1],
            color=sns.color_palette("viridis", len(imp)))
    ax.set_title("Feature importance - best learner")
    ax.set_xlabel("importance")
    fig.suptitle("Machine learning - lagged features, chronological split",
                 fontsize=14, fontweight="bold")
    save(fig, numbered(12, "ml_results"))


def fig_ml_tuning(res):
    names = [k for k in res["results"] if k != "Naive (tomorrow = today)"]
    cv = [res["results"][k].get("CV_RMSE", np.nan) for k in names]
    test_rmse = [res["results"][k]["RMSE"] for k in names]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names))
    ax.bar(x - .2, cv, width=.4, label="CV RMSE (TimeSeriesSplit)", color=MAIN)
    ax.bar(x + .2, test_rmse, width=.4, label="hold-out RMSE", color=WARN)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("RMSE")
    ax.set_title("Cross-validated vs hold-out error")
    ax.legend(frameon=False)
    save(fig, numbered(13, "ml_tuning_comparison"))


def fig_arima(ar):
    s = ar["series"]
    sm = ar["summary"]
    fc = ar["forecast"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    ax = axes[0]
    ax.plot(s.index, s.values, color=MAIN, lw=1.5, label="observed")
    ax.plot(ar["test"].index, ar["pred"].values, color=WARN, lw=2,
            label=f"ARIMA{sm['order']} hold-out forecast")
    ax.fill_between(ar["test"].index, ar["conf"].iloc[:, 0], ar["conf"].iloc[:, 1],
                    color=WARN, alpha=.2)
    ax.set_title("Hold-out validation (final 21 days)")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1]
    ax.plot(s.index, s.values, color=MAIN, lw=1.5, label="observed")
    ax.plot(fc["Date"], fc["Forecast"], color=WARN, lw=2, label="30-day forecast")
    ax.fill_between(fc["Date"], fc["Lower_95"], fc["Upper_95"], color=WARN, alpha=.2,
                    label="95% interval")
    ax.axvline(s.index.max(), color="grey", ls="--", lw=1)
    ax.set_title("Forward forecast: 04 Jun - 03 Jul 2020")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"ARIMA{sm['order']} - validation and 30-day projection",
                 fontsize=14, fontweight="bold")
    save(fig, numbered(14, "arima_forecast"))


def fig_stats(out):
    from scipy import stats
    daily = out["daily"]
    df = out["df"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    stats.probplot(daily["Cases"], dist="norm", plot=ax)
    ax.set_title("Q-Q plot: daily cases vs normal")

    ax = axes[1]
    prov = out["province"]
    ax.bar(prov["Province"], prov["Recovery_Rate"], color=sns.color_palette(PALETTE, len(prov)))
    ax.set_ylabel("recovery %")
    ax.set_title("Recovery rate by province")
    ax.tick_params(axis="x", rotation=45)

    ax = axes[2]
    sns.violinplot(data=df, x="Is_Local", y="Cases", ax=ax, hue="Is_Local",
                   palette="Set2", legend=False, cut=0)
    ax.set_yscale("symlog")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["imported / unknown", "local transmission"])
    ax.set_title("Record cases: local vs imported")
    save(fig, numbered(15, "statistics_panel"))


def fig_weekly(out):
    df = out["df"]
    wk = (df.groupby("Week")
            .agg(Cases=("Cases", "sum"), Deaths=("Deaths", "sum"))
            .reset_index())
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(wk["Week"], wk["Cases"], color=MAIN, alpha=.85, label="cases")
    ax2 = ax.twinx()
    ax2.plot(wk["Week"], wk["Deaths"], color=WARN, lw=2, marker="o", ms=3, label="deaths")
    ax.set_xlabel("ISO week")
    ax.set_ylabel("cases")
    ax2.set_ylabel("deaths")
    ax.set_title("Weekly cases and deaths")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    save(fig, numbered(16, "weekly_trend"))


def fig_dayofweek(out):
    daily = out["daily"]
    daily = daily.copy()
    daily["dow"] = daily["Date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=daily, x="dow", y="Cases", order=order, ax=ax,
                hue="dow", palette="viridis", legend=False)
    ax.set_title("Mean daily cases by day of week")
    ax.set_ylabel("mean cases")
    save(fig, numbered(17, "day_of_week"))


def fig_top_cities_full(out):
    city = out["city"].head(20)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("viridis", len(city))
    ax.barh(city["City"][::-1], city["Cases"][::-1], color=colors[::-1])
    ax.set_title("Top 20 cities by total cases")
    ax.set_xlabel("cases")
    for y, v in enumerate(city["Cases"][::-1]):
        ax.text(v, y, f" {v:,.0f}", va="center", fontsize=7)
    save(fig, numbered(18, "top20_cities"))


def fig_master(out, res, ar):
    """One-page summary image for the README."""
    daily = out["daily"]
    prov = out["province"]
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, hspace=.45, wspace=.3)

    ax = fig.add_subplot(gs[0, :2])
    ax.plot(daily["Date"], daily["Cases"], color="#9aa5b1", lw=1, label="daily")
    ax.plot(daily["Date"], daily["MA7_Cases"], color=MAIN, lw=2.4, label="7-day mean")
    ax.set_title("Daily cases")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.tick_params(axis="x", rotation=30)

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    kpis = [
        ("Total cases", f"{int(out['df'].Cases.sum()):,}"),
        ("Deaths", f"{int(out['df'].Deaths.sum()):,}"),
        ("Recovered", f"{int(out['df'].Recovered.sum()):,}"),
        ("CFR", f"{out['df'].Deaths.sum() / out['df'].Cases.sum() * 100:.2f}%"),
        ("Cities", f"{out['report'].cities_final}"),
        ("Province", "8"),
    ]
    for i, (k, v) in enumerate(kpis):
        ax.text(.02, .92 - i * .155, k, fontsize=10, color="#555")
        ax.text(.02, .87 - i * .155, v, fontsize=15, fontweight="bold", color=NAVY if False else "#1F3864")

    ax = fig.add_subplot(gs[1, 0])
    ax.barh(prov["Province"][::-1], prov["Cases"][::-1], color=sns.color_palette(PALETTE, len(prov)))
    ax.set_title("Cases by province", fontsize=10)
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(daily["Date"], daily["CFR"], color=WARN, lw=2)
    ax.plot(daily["Date"], daily["Recovery_Rate"], color="#375623", lw=2)
    ax.set_title("CFR vs recovery %", fontsize=10)
    ax.tick_params(labelsize=7, axis="x", rotation=30)

    ax = fig.add_subplot(gs[1, 2])
    t = out["travel"].head(6)
    ax.pie(t["Cases"], labels=[s[:12] for s in t["Travel_history"]], autopct="%1.0f%%",
           colors=sns.color_palette(PALETTE, len(t)), textprops={"fontsize": 7})
    ax.set_title("Exposure source", fontsize=10)

    ax = fig.add_subplot(gs[2, 0])
    city = out["city"].head(8)
    ax.barh(city["City"][::-1], city["Cases"][::-1], color=sns.color_palette("magma", len(city)))
    ax.set_title("Top cities", fontsize=10)
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[2, 1:])
    fc = ar["forecast"]
    ax.plot(daily["Date"].tail(40), daily["Cases"].tail(40), color=MAIN, lw=1.6, label="observed")
    ax.plot(fc["Date"], fc["Forecast"], color=WARN, lw=2, label="ARIMA forecast")
    ax.fill_between(fc["Date"], fc["Lower_95"], fc["Upper_95"], color=WARN, alpha=.2)
    ax.set_title("30-day ARIMA projection", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(labelsize=7, axis="x", rotation=30)

    fig.suptitle("COVID-19 Pakistan - master dashboard", fontsize=17, fontweight="bold")
    save(fig, numbered(19, "master_dashboard"))


# ---------------------------------------------------------- interactive
def interactive(out, ar):
    import plotly.express as px
    import plotly.graph_objects as go

    daily = out["daily"].copy()
    daily["Date"] = pd.to_datetime(daily["Date"])

    # 1. daily trend
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Cases"], name="Daily cases",
                             line=dict(color="#2E75B6", width=1.5)))
    fig.add_trace(go.Scatter(x=daily["Date"], y=daily["MA7_Cases"], name="7-day average",
                             line=dict(color="#BF8F00", width=2.5)))
    fig.update_layout(title="Daily COVID-19 cases in Pakistan",
                      template="plotly_white", height=520,
                      xaxis_title="date", yaxis_title="cases")
    fig.write_html(os.path.join(INT_DIR, "01_daily_trend.html"),
                   include_plotlyjs="cdn", full_html=True)
    print("  saved 01_daily_trend.html")

    # 2. province bubble
    prov = out["province"]
    fig = px.scatter(prov, x="Cases", y="CFR", size="Active", color="Province",
                     hover_name="Province", size_max=70, log_x=True,
                     labels={"Cases": "total cases (log)", "CFR": "case fatality %"},
                     title="Province footprint: cases, fatality and active cases")
    fig.update_layout(template="plotly_white", height=560)
    fig.write_html(os.path.join(INT_DIR, "02_province_bubble.html"),
                   include_plotlyjs="cdn", full_html=True)
    print("  saved 02_province_bubble.html")

    # 3. treemap
    city = out["city"]
    fig = px.treemap(city, path=["Province", "City"], values="Cases",
                     color="CFR", color_continuous_scale="YlOrRd",
                     title="Case load by province and city")
    fig.update_layout(template="plotly_white", height=620)
    fig.write_html(os.path.join(INT_DIR, "03_treemap.html"),
                   include_plotlyjs="cdn", full_html=True)
    print("  saved 03_treemap.html")

    # 4. sunburst of exposure source
    df = out["df"]
    agg = (df.groupby(["Travel_history", "Province"])["Cases"].sum().reset_index())
    fig = px.sunburst(agg, path=["Travel_history", "Province"], values="Cases",
                      title="Exposure source and province")
    fig.update_layout(template="plotly_white", height=620)
    fig.write_html(os.path.join(INT_DIR, "04_sunburst.html"),
                   include_plotlyjs="cdn", full_html=True)
    print("  saved 04_sunburst.html")

    # 5. province x month heatmap
    piv = df.pivot_table(values="Cases", index="Province", columns="Month_Short",
                         aggfunc="sum", fill_value=0)
    months = [m for m in ["Feb", "Mar", "Apr", "May", "Jun"] if m in piv.columns]
    fig = go.Figure(go.Heatmap(z=piv[months].values, x=months, y=piv.index,
                               colorscale="YlOrRd", text=piv[months].values,
                               texttemplate="%{text:,.0f}"))
    fig.update_layout(title="Cases by province and month", template="plotly_white",
                      height=520, yaxis_autorange="reversed")
    fig.write_html(os.path.join(INT_DIR, "05_heatmap.html"),
                   include_plotlyjs="cdn", full_html=True)
    print("  saved 05_heatmap.html")

    # 6. animated race
    cum = (df.groupby(["Province", "Date"])["Cases"].sum().reset_index()
             .sort_values("Date"))
    cum["cum"] = cum.groupby("Province")["Cases"].cumsum()
    fig = px.bar(cum, x="Province", y="cum", color="Province",
                 animation_frame="Date", range_y=[0, cum["cum"].max() * 1.05],
                 title="Cumulative cases by province (animated)")
    fig.update_layout(template="plotly_white", height=560, showlegend=False)
    fig.write_html(os.path.join(INT_DIR, "06_animated.html"),
                   include_plotlyjs="cdn", full_html=True)
    print("  saved 06_animated.html")

    for f in sorted(os.listdir(INT_DIR)):
        kb = os.path.getsize(os.path.join(INT_DIR, f)) / 1024
        print("     %-26s %6.1f KB" % (f, kb))


# ------------------------------------------------------------------ run
def main():
    print("Building figures ...")
    out = load()
    res = train_and_evaluate(verbose=False)
    from arima_forecast import run as arima_run
    ar = arima_run(verbose=False)

    fig_overview(out)
    fig_daily_trends(out["daily"])
    fig_cumulative_rates(out["daily"])
    fig_province_city(out)
    fig_province_cfr(out["province"])
    fig_monthly_heatmap(out)
    fig_travel(out)
    fig_distributions(out)
    fig_correlation(out["daily"], out["df"])
    fig_scatter(out["daily"], out["df"])
    fig_ml(res, out)
    fig_ml_tuning(res)
    fig_arima(ar)
    fig_stats(out)
    fig_weekly(out)
    fig_dayofweek(out)
    fig_top_cities_full(out)
    fig_master(out, res, ar)

    print("\nInteractive charts ...")
    interactive(out, ar)
    print("\n%d static figures written to %s" % (len(_saved), FIG_DIR))
    return out, res, ar


if __name__ == "__main__":
    main()
