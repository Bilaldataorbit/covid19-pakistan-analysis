"""
src/reports.py
==============
Generates the two deliverables the README advertises:

* outputs/COVID19_Analysis_Report.xlsx  - multi-sheet Excel workbook
* outputs/COVID19_Insights_Report.txt   - plain-text executive summary

Every number comes from the same pipeline, so the report can never drift out of
sync with the data (the previous README quoted 83,986 cases when the data says 83,985,
and listed Lahore at 15,000 when it is 28,015).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from arima_forecast import run as arima_run
from data_pipeline import ROOT, build_all
from ml_models import feature_importance, train_and_evaluate

OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

NAVY = "1F3864"
HDR = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
FILL = PatternFill("solid", fgColor=NAVY)
TITLE = Font(name="Calibri", size=15, bold=True, color=NAVY)
BOLD = Font(name="Calibri", size=10, bold=True)
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _sheet(wb, title, df, start=3, note=None):
    ws = wb.create_sheet(title[:31])
    ws.cell(row=1, column=1, value=title).font = TITLE
    r = start
    if note:
        ws.cell(row=2, column=1, value=note).font = Font(name="Calibri", size=9,
                                                         italic=True, color="808080")
    for j, col in enumerate(df.columns, 1):
        c = ws.cell(row=r, column=j, value=str(col))
        c.font = HDR; c.fill = FILL; c.border = BOX
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, row in enumerate(df.itertuples(index=False), start=r + 1):
        for j, v in enumerate(row, 1):
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            c = ws.cell(row=i, column=j, value=v)
            c.border = BOX
            c.font = Font(name="Calibri", size=9)
    for j in range(1, len(df.columns) + 1):
        width = max(12, min(30, int(df[df.columns[j - 1]].astype(str).str.len().max() or 12) + 2))
        ws.column_dimensions[get_column_letter(j)].width = width
    ws.freeze_panes = ws.cell(row=r + 1, column=1)
    return ws


def build_excel(out, res, ar, path=None) -> str:
    path = path or os.path.join(OUT_DIR, "COVID19_Analysis_Report.xlsx")
    df, daily, prov, city, travel = (out["df"], out["daily"], out["province"],
                                     out["city"], out["travel"])
    rep = out["report"]

    wb = Workbook()
    wb.remove(wb.active)

    # 1. summary
    ws = wb.create_sheet("Summary")
    ws["A1"] = "COVID-19 Pakistan - analysis report"
    ws["A1"].font = Font(name="Calibri", size=17, bold=True, color=NAVY)
    ws["A2"] = "Generated from data/processed by src/reports.py"
    ws["A2"].font = Font(name="Calibri", size=9, italic=True, color="808080")

    rows = [
        ("Dataset", ""),
        ("Raw rows", rep.raw_rows),
        ("Rows after de-duplication", rep.after_exact_dupes),
        ("Final analytical rows", rep.final_rows),
        ("Date range", f"{df.Date.min().date()} to {df.Date.max().date()}"),
        ("Provinces", rep.provinces_final),
        ("Cities (normalised)", rep.cities_final),
        ("", ""),
        ("Headline totals", ""),
        ("Total cases", rep.total_cases),
        ("Total deaths", rep.total_deaths),
        ("Total recovered", rep.total_recovered),
        ("Active cases", rep.total_cases - rep.total_deaths - rep.total_recovered),
        ("Case fatality rate %", round(rep.total_deaths / rep.total_cases * 100, 3)),
        ("Recovery rate %", round(rep.total_recovered / rep.total_cases * 100, 3)),
        ("", ""),
        ("Data quality notes", ""),
        ("Exact duplicate rows removed", rep.raw_rows - rep.after_exact_dupes),
        ("City spellings merged", rep.cities_raw - rep.cities_final),
        ("Rows with blank Travel_history", rep.missing_travel_rows),
        ("Cases in those rows", rep.missing_travel_cases),
        ("Their share of all cases %", round(rep.missing_travel_cases / rep.total_cases * 100, 2)),
        ("", ""),
        ("Modelling", ""),
        ("ML split", "chronological, final 21 days held out"),
        ("ML features", "14 lagged features (no same-day predictors)"),
        ("Best learner", res["best_name"]),
        ("Naive benchmark RMSE", round(res["benchmark"]["RMSE"], 1)),
        ("Best learner RMSE", round(res["results"][res["best_name"]]["RMSE"], 1)),
        ("ARIMA order", str(ar["summary"]["order"])),
        ("ARIMA hold-out MAE", round(ar["holdout_metrics"]["MAE"], 1)),
        ("ARIMA 30-day forecast total", ar["summary"]["forecast_total"]),
    ]
    r = 4
    for k, v in rows:
        if v == "" and k:
            c = ws.cell(row=r, column=1, value=k)
            c.font = Font(name="Calibri", size=11, bold=True, color=NAVY)
            c.fill = PatternFill("solid", fgColor="DEEBF7")
            ws.cell(row=r, column=2).fill = PatternFill("solid", fgColor="DEEBF7")
        elif k:
            ws.cell(row=r, column=1, value=k).font = BOLD
            c = ws.cell(row=r, column=2, value=v)
            c.font = Font(name="Calibri", size=10)
            c.alignment = Alignment(horizontal="left")
        r += 1
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 34

    # 2..n data sheets
    _sheet(wb, "Clean Data", df.head(3000),
           note=f"{len(df)} rows after cleaning (first 3000 shown)")
    _sheet(wb, "Daily", daily, note="One row per calendar day, gap days included as zero")
    _sheet(wb, "Province", prov)
    _sheet(wb, "City", city, note="City names normalised - duplicates merged")
    _sheet(wb, "Travel", travel,
           note=f"Unknown includes {rep.missing_travel_rows} rows whose source Travel_history was blank")

    # ML results
    ml_rows = []
    for name, m in res["results"].items():
        ml_rows.append({
            "Model": name,
            "R2": round(m["R2"], 4),
            "RMSE": round(m["RMSE"], 1),
            "MAE": round(m["MAE"], 1),
            "MAPE %": round(m["MAPE"], 1),
            "CV RMSE": round(m.get("CV_RMSE", float("nan")), 1) if m.get("CV_RMSE") == m.get("CV_RMSE") else "",
        })
    _sheet(wb, "ML Results", pd.DataFrame(ml_rows),
           note="Hold-out = final 21 days. Naive row is the benchmark every model must beat.")

    params = []
    for name, p in res["best_params"].items():
        params.append({"Model": name, "Best params": json.dumps(p) if p else "not tuned"})
    _sheet(wb, "ML Tuning", pd.DataFrame(params),
           note="GridSearchCV with TimeSeriesSplit(4) on the training window")

    imp = feature_importance(res["best_model"], res["features"])
    if len(imp):
        _sheet(wb, "Feature Importance", imp)

    # ARIMA sheets
    _sheet(wb, "ARIMA Holdout", pd.DataFrame({
        "Date": ar["test"].index,
        "Actual": ar["test"].values,
        "Predicted": np.round(ar["pred"].values, 0),
        "Lower_95": np.round(ar["conf"].iloc[:, 0].values, 0),
        "Upper_95": np.round(ar["conf"].iloc[:, 1].values, 0),
    }), note=f"ARIMA{ar['summary']['order']} refit on the first 78 days")
    _sheet(wb, "ARIMA Forecast", ar["forecast"],
           note="30-day projection with a 95% interval")
    _sheet(wb, "ARIMA Order Grid", ar["grid"].head(20),
           note="AIC grid search used to pick the order")

    wb.save(path)
    print("  wrote", path)
    return path


def build_txt(out, res, ar, path=None) -> str:
    path = path or os.path.join(OUT_DIR, "COVID19_Insights_Report.txt")
    df, daily, prov, city, travel = (out["df"], out["daily"], out["province"],
                                     out["city"], out["travel"])
    rep = out["report"]
    L = []
    add = L.append

    add("=" * 78)
    add("COVID-19 PAKISTAN - INSIGHTS REPORT")
    add("=" * 78)
    add("")
    add("DATASET")
    add("-" * 78)
    add(f"  Raw rows                 : {rep.raw_rows:,}")
    add(f"  Rows after de-duplication: {rep.after_exact_dupes:,}")
    add(f"  Final analytical rows    : {rep.final_rows:,}")
    add(f"  Date range               : {df.Date.min().date()} to {df.Date.max().date()}")
    add(f"  Provinces                : {rep.provinces_final}")
    add(f"  Cities (normalised)      : {rep.cities_final}  (from {rep.cities_raw} raw spellings)")
    add("")
    add("HEADLINE NUMBERS")
    add("-" * 78)
    add(f"  Total cases      : {rep.total_cases:,}")
    add(f"  Total deaths     : {rep.total_deaths:,}")
    add(f"  Total recovered  : {rep.total_recovered:,}")
    add(f"  Active cases     : {rep.total_cases - rep.total_deaths - rep.total_recovered:,}")
    add(f"  Case fatality %  : {rep.total_deaths / rep.total_cases * 100:.2f}")
    add(f"  Recovery rate %  : {rep.total_recovered / rep.total_cases * 100:.2f}")
    add("")
    add("DATA QUALITY")
    add("-" * 78)
    add(f"  Exact duplicates removed : {rep.raw_rows - rep.after_exact_dupes}")
    add(f"  City spellings merged    : {rep.cities_raw - rep.cities_final}")
    add(f"  Rows with blank travel   : {rep.missing_travel_rows}")
    add(f"  Cases in those rows      : {rep.missing_travel_cases:,}"
        f"  ({rep.missing_travel_cases / rep.total_cases * 100:.2f}% of all cases)")
    add("  These rows were labelled 'Unknown'. They are the single largest")
    add("  reporting gap in the dataset and they are concentrated in Punjab")
    add("  (Lahore) and Balochistan, so 'Unknown' should be read as 'not stated',")
    add("  not as a transmission category.")
    add("")
    add("PROVINCES")
    add("-" * 78)
    for _, r in prov.iterrows():
        add("  %-34s cases=%9s  deaths=%6s  CFR=%5.2f%%  share=%5.2f%%"
            % (r["Province"], f"{r['Cases']:,}", f"{r['Deaths']:,}", r["CFR"], r["Case_Share"]))
    add("")
    add("TOP 10 CITIES")
    add("-" * 78)
    for i, (_, r) in enumerate(city.head(10).iterrows(), 1):
        add("  %2d. %-18s %-28s cases=%9s  CFR=%5.2f%%"
            % (i, r["City"], r["Province"], f"{r['Cases']:,}", r["CFR"]))
    add("")
    add("EXPOSURE SOURCE")
    add("-" * 78)
    for _, r in travel.head(8).iterrows():
        add("  %-26s cases=%9s  share=%5.2f%%  records=%5d"
            % (r["Travel_history"], f"{r['Cases']:,}", r["Case_Share"], r["Records"]))
    add("")
    add("DAILY SERIES")
    add("-" * 78)
    peak = daily.loc[daily["Cases"].idxmax()]
    add(f"  Days covered        : {len(daily)} calendar days ({int((daily['Cases']==0).sum())} with zero new cases)")
    add(f"  Mean daily cases    : {daily['Cases'].mean():.1f}")
    add(f"  Median daily cases  : {daily['Cases'].median():.0f}")
    add(f"  Peak day            : {peak['Date'].date()} with {int(peak['Cases']):,} cases")
    add(f"  Last 7-day average  : {daily['MA7_Cases'].iloc[-1]:.1f}")
    add("")
    add("MACHINE LEARNING (chronological hold-out, lagged features only)")
    add("-" * 78)
    add("  Every predictor is lagged by at least one day, so no same-day")
    add("  information leaks into the target.")
    add("")
    add("  %-24s %8s %8s %8s %8s" % ("model", "R2", "RMSE", "MAE", "MAPE%"))
    for name, m in res["results"].items():
        add("  %-24s %8.3f %8.1f %8.1f %8.1f"
            % (name, m["R2"], m["RMSE"], m["MAE"], m["MAPE"]))
    add("")
    add(f"  Best learner: {res['best_name']}")
    add(f"  Naive benchmark RMSE: {res['benchmark']['RMSE']:.1f}")
    beat = res["results"][res["best_name"]]["RMSE"] < res["benchmark"]["RMSE"]
    if beat:
        add("  The tuned model beats the naive benchmark on the hold-out.")
    else:
        add("  The tuned model does NOT beat the naive benchmark on the hold-out.")
        add("  This is the honest result: with 62 training days and a steeply rising")
        add("  curve, lag-1 is a very strong predictor and the extra features add")
        add("  noise rather than signal. The earlier R2 of 0.75 came from using")
        add("  same-row Deaths/Recovered and a random split, which is leakage.")
    add("")
    add("ARIMA FORECAST")
    add("-" * 78)
    s = ar["summary"]
    add(f"  Selected order     : ARIMA{s['order']}   (AIC={s['aic']:.1f}, BIC={s['bic']:.1f})")
    add(f"  Hold-out MAE       : {ar['holdout_metrics']['MAE']:.1f}")
    add(f"  Hold-out RMSE      : {ar['holdout_metrics']['RMSE']:.1f}")
    add(f"  Hold-out MAPE      : {ar['holdout_metrics']['MAPE']:.1f}%")
    add(f"  Naive hold-out MAE : {s['naive_holdout_mae']:.1f}")
    add(f"  Seasonal naive MAE : {s['seasonal_naive_holdout_mae']:.1f}")
    add("")
    add(f"  30-day forecast total : {s['forecast_total']:,} cases")
    add(f"  Mean daily            : {s['forecast_mean_daily']:,.0f}")
    add(f"  Peak day              : {s['forecast_peak_day']:,} on {s['forecast_peak_date']}")
    add(f"  Minimum day           : {s['forecast_min_day']:,}")
    add("")
    add("READING NOTES")
    add("-" * 78)
    add("  * CFR and recovery rate are cumulative ratios; they are unstable in the")
    add("    first weeks when the denominator is tiny.")
    add("  * Recovered is under-reported, so the recovery rate is a lower bound.")
    add("  * Exposure source and province are correlated, so provincial CFR")
    add("    differences are not a clean measure of care quality.")
    add("  * Forecasts are scenario envelopes, not predictions.")
    add("")
    add("=" * 78)

    text = "\n".join(L)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("  wrote", path)
    return path


def main():
    print("Building reports ...")
    out = build_all(verbose=False)
    res = train_and_evaluate(verbose=False)
    ar = arima_run(verbose=False)
    p1 = build_excel(out, res, ar)
    p2 = build_txt(out, res, ar)
    return p1, p2


if __name__ == "__main__":
    main()
