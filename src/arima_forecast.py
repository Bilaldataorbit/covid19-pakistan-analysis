"""
src/arima_forecast.py
=====================
Time-series forecasting of daily COVID-19 cases with ARIMA.

The earlier version reported a MAPE of 23% and a 30-day forecast of ~196k
additional cases. This module re-estimates the model properly:

* the daily series is rebuilt from the cleaned pipeline, so gap days are
  present as zeros instead of being silently skipped;
* the order is chosen by a small grid search over AIC, then validated on a
  chronological hold-out (the final 21 days), which is the only honest way
  to quote an error on a time series;
* the reported MAPE is the hold-out MAPE, not a fitted-value statistic.

Outputs: outputs/arima_*.csv, models/arima_model.pkl
"""
from __future__ import annotations

import itertools
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from data_pipeline import ROOT, build_all

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(ROOT, "outputs")
MODELS_DIR = os.path.join(ROOT, "models")
TEST_DAYS = 21
HORIZON = 30


def order_search(series: pd.Series, p_range=(0, 1, 2, 3, 7), d_range=(0, 1, 2), q_range=(0, 1, 2, 3)):
    """Small AIC grid - the standard starting point before a manual diagnostic."""
    best = None
    rows = []
    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            m = ARIMA(series, order=(p, d, q)).fit()
            rows.append({"p": p, "d": d, "q": q, "AIC": m.aic, "BIC": m.bic})
            if best is None or m.aic < best["AIC"]:
                best = {"p": p, "d": d, "q": q, "AIC": m.aic, "BIC": m.bic,
                        "model": m}
        except Exception:
            continue
    return best, pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = np.maximum(np.abs(y_true), 1.0)
    mape = np.mean(np.abs(y_true - y_pred) / denom) * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}


def run(verbose: bool = True):
    out = build_all(verbose=False)
    daily = out["daily"][["Date", "Cases"]].copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values("Date").reset_index(drop=True)
    s = daily.set_index("Date")["Cases"].astype(float)

    # stationarity diagnostic
    adf = adfuller(s.dropna(), autolag="AIC")
    if verbose:
        print("Daily series: %d rows, %s -> %s" % (len(s), s.index.min().date(), s.index.max().date()))
        print("ADF on levels   : stat=%.3f  p=%.4f" % (adf[0], adf[1]))
        adf_d = adfuller(s.diff().dropna(), autolag="AIC")
        print("ADF on 1st diff : stat=%.3f  p=%.4f" % (adf_d[0], adf_d[1]))

    train, test = s.iloc[:-TEST_DAYS], s.iloc[-TEST_DAYS:]
    order_global, grid = order_search(train)
    order = (order_global["p"], order_global["d"], order_global["q"])
    if verbose:
        print("\nSelected order:", order, "| AIC=%.2f BIC=%.2f"
              % (order_global["AIC"], order_global["BIC"]))
        print(grid.head(6).to_string(index=False))

    # hold-out evaluation: refit on train, forecast the last 21 days
    holdout = ARIMA(train, order=order).fit()
    fc = holdout.get_forecast(steps=TEST_DAYS)
    pred = fc.predicted_mean
    conf = fc.conf_int(alpha=0.05)
    hold_metrics = metrics(test.values, pred.values)

    # naive benchmark: predict each test day with the previous day's actual value
    full = s.values
    idx_test_start = len(full) - TEST_DAYS
    naive_pred = full[idx_test_start - 1: -1]
    naive_mae = float(np.mean(np.abs(test.values - naive_pred)))
    seasonal_pred = full[idx_test_start - 7: len(full) - 7]
    seasonal_mae = float(np.mean(np.abs(test.values - seasonal_pred)))
    if verbose:
        print("\nHold-out (%d days): MAE=%.1f RMSE=%.1f MAPE=%.1f%%"
              % (TEST_DAYS, hold_metrics["MAE"], hold_metrics["RMSE"], hold_metrics["MAPE"]))
        print("Naive (yesterday) MAE=%.1f  |  seasonal naive (t-7) MAE=%.1f" % (naive_mae, seasonal_mae))
        verdict = []
        if hold_metrics["MAE"] < naive_mae:
            verdict.append("beats naive")
        else:
            verdict.append("does NOT beat naive")
        if hold_metrics["MAE"] < seasonal_mae:
            verdict.append("beats seasonal naive")
        else:
            verdict.append("does NOT beat seasonal naive")
        print("Verdict: ARIMA " + " and ".join(verdict))

    # final model on the full series, then the 30-day forward forecast
    final = ARIMA(s, order=order).fit()
    fwd = final.get_forecast(steps=HORIZON)
    mean = fwd.predicted_mean
    ci = fwd.conf_int(alpha=0.05)
    future_dates = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    forecast = pd.DataFrame({
        "Date": future_dates,
        "Forecast": np.round(mean.values, 0).astype(int),
        "Lower_95": np.round(ci.iloc[:, 0].values, 0).astype(int),
        "Upper_95": np.round(ci.iloc[:, 1].values, 0).astype(int),
    })
    forecast.loc[forecast["Lower_95"] < 0, "Lower_95"] = 0

    summary = {
        "order": order,
        "aic": float(order_global["AIC"]),
        "bic": float(order_global["BIC"]),
        "adf_levels_p": float(adf[1]),
        "holdout_metrics": hold_metrics,
        "naive_holdout_mae": float(naive_mae),
        "seasonal_naive_holdout_mae": seasonal_mae,
        "horizon_days": HORIZON,
        "forecast_total": int(forecast["Forecast"].sum()),
        "forecast_mean_daily": float(forecast["Forecast"].mean()),
        "forecast_peak_day": int(forecast["Forecast"].max()),
        "forecast_peak_date": str(forecast.loc[forecast["Forecast"].idxmax(), "Date"].date()),
        "forecast_min_day": int(forecast["Forecast"].min()),
        "observed_total": int(s.sum()),
        "observed_last_date": str(s.index.max().date()),
    }
    if verbose:
        print("\nFinal ARIMA%s on the full series" % (order,))
        print("  30-day forecast total : %s" % f"{summary['forecast_total']:,}")
        print("  mean daily            : %s" % f"{summary['forecast_mean_daily']:,.0f}")
        print("  peak day              : %s on %s" %
              (f"{summary['forecast_peak_day']:,}", summary["forecast_peak_date"]))
        print("  range                 : %s - %s"
              % (f"{summary['forecast_min_day']:,}", f"{summary['forecast_peak_day']:,}"))

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    forecast.to_csv(os.path.join(OUT_DIR, "arima_forecast_30d.csv"), index=False)
    grid.to_csv(os.path.join(OUT_DIR, "arima_order_grid.csv"), index=False)
    pd.DataFrame({
        "Date": test.index,
        "Actual": test.values,
        "Predicted": np.round(pred.values, 0),
        "Lower_95": np.round(conf.iloc[:, 0].values, 0),
        "Upper_95": np.round(conf.iloc[:, 1].values, 0),
    }).to_csv(os.path.join(OUT_DIR, "arima_holdout.csv"), index=False)
    with open(os.path.join(OUT_DIR, "arima_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(MODELS_DIR, "arima_model.pkl"), "wb") as f:
        pickle.dump({"order": order, "result": final}, f)

    return dict(summary=summary, forecast=forecast, holdout_metrics=hold_metrics,
                grid=grid, train=train, test=test, pred=pred, conf=conf,
                series=s, fit=final)


if __name__ == "__main__":
    run()
