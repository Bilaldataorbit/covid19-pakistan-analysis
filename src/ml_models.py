"""
src/ml_models.py
================
Leakage-free forecasting of daily COVID-19 cases.

Why the previous model was wrong
--------------------------------
* it used `Deaths` and `Recovered` from the SAME row as the target `Cases`,
  so the model was allowed to see information that would not exist at
  prediction time;
* it used a random train/test split on a time series, so test days sat
  between training days and the score was inflated;
* GridSearchCV was imported but never used, and the README still claimed
  tuning had been done.

What this module does instead
-----------------------------
* predictors are all lagged: lag-1/3/7 cases, lag-1 deaths and recoveries,
  rolling mean/std/max computed on t-1, cumulative cases up to t-1;
* the split is chronological - the last 21 days are held out;
* four models are compared, and Random Forest / Gradient Boosting are
  tuned with TimeSeriesSplit inside GridSearchCV;
* metrics are reported on the held-out tail, next to a naive
  "tomorrow = today" benchmark, because that is what a forecast has to beat.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline import (DAILY_PATH, MODEL_FEATURES, ROOT, TARGET,
                           build_all, model_table)

MODELS_DIR = os.path.join(ROOT, "models")
TEST_DAYS = 21          # chronological hold-out
SEED = 42


def load_model_table():
    out = build_all(verbose=False)
    return out["model"], out["daily"], out["report"]


def time_split(df: pd.DataFrame, test_days: int = TEST_DAYS):
    """Chronological split: the final `test_days` rows are the test set."""
    df = df.sort_values("Date").reset_index(drop=True)
    cut = len(df) - test_days
    train, test = df.iloc[:cut], df.iloc[cut:]
    return train, test


def metrics(y_true, y_pred) -> dict:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred))
                                    / np.maximum(np.asarray(y_true), 1))) * 100),
    }


def naive_benchmark(test: pd.DataFrame) -> dict:
    """Tomorrow = today. Any useful model must beat this."""
    return metrics(test[TARGET].values, test["Lag_Cases_1"].values)


def build_candidates() -> dict:
    tscv = TimeSeriesSplit(n_splits=4)
    return {
        "Linear Regression": (
            Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            {},
        ),
        "Ridge Regression": (
            Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=SEED))]),
            {"model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=SEED, n_jobs=1),
            {
                "n_estimators": [200, 400],
                "max_depth": [3, 4, 6, None],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.5, 1.0],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=SEED),
            {
                "n_estimators": [150, 300],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.8, 1.0],
            },
        ),
    }


def train_and_evaluate(verbose: bool = True):
    model_df, daily, report = load_model_table()
    train, test = time_split(model_df)
    X_train, y_train = train[MODEL_FEATURES], train[TARGET]
    X_test, y_test = test[MODEL_FEATURES], test[TARGET]

    if verbose:
        print("Train rows: %d  (%s -> %s)" % (len(train), train.Date.min().date(), train.Date.max().date()))
        print("Test rows : %d  (%s -> %s)" % (len(test), test.Date.min().date(), test.Date.max().date()))
        print("Features  : %d  ->  %s" % (len(MODEL_FEATURES), MODEL_FEATURES))
        print()

    results, fitted, best_params = {}, {}, {}
    for name, (estimator, grid) in build_candidates().items():
        if grid:
            gs = GridSearchCV(estimator, grid, cv=TimeSeriesSplit(n_splits=4),
                              scoring="neg_root_mean_squared_error",
                              n_jobs=1, refit=True)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            best_params[name] = gs.best_params_
            cv_rmse = -gs.best_score_
        else:
            best = estimator
            best.fit(X_train, y_train)
            best_params[name] = {}
            cv_rmse = float("nan")

        pred = best.predict(X_test)
        m = metrics(y_test.values, pred)
        m["CV_RMSE"] = float(cv_rmse)
        results[name] = m
        fitted[name] = best
        if verbose:
            print("%-20s R2=%6.3f  RMSE=%7.1f  MAE=%6.1f  MAPE=%5.1f%%  CV_RMSE=%7.1f"
                  % (name, m["R2"], m["RMSE"], m["MAE"], m["MAPE"], m["CV_RMSE"]))

    bench = naive_benchmark(test)
    results["Naive (tomorrow = today)"] = bench
    fitted["Naive (tomorrow = today)"] = None
    best_params["Naive (tomorrow = today)"] = {}
    if verbose:
        print("%-20s R2=%6.3f  RMSE=%7.1f  MAE=%6.1f  MAPE=%5.1f%%"
              % ("Naive benchmark", bench["R2"], bench["RMSE"], bench["MAE"], bench["MAPE"]))
        print()

    # the best model is the one with the lowest test RMSE among the real learners
    real = {k: v for k, v in results.items() if k != "Naive (tomorrow = today)"}
    best_name = min(real, key=lambda k: real[k]["RMSE"])
    if verbose:
        print("Best model on the hold-out:", best_name)
        beat = real[best_name]["RMSE"] < bench["RMSE"]
        print("Beats the naive benchmark:", "YES" if beat else "NO")
        print("Best hyperparameters:", json.dumps(best_params[best_name], indent=2))

    return dict(
        results=results, best_name=best_name, best_model=fitted[best_name],
        best_params=best_params, train=train, test=test,
        y_pred=fitted[best_name].predict(X_test),
        benchmark=bench, features=MODEL_FEATURES, report=report,
    )


def feature_importance(fitted_model, features) -> pd.DataFrame:
    """Return permutation-free built-in importances when available."""
    mdl = fitted_model
    if isinstance(mdl, Pipeline):
        mdl = mdl.named_steps["model"]
    if hasattr(mdl, "feature_importances_"):
        imp = mdl.feature_importances_
        how = "impurity importance"
    elif hasattr(mdl, "coef_"):
        imp = np.abs(np.ravel(mdl.coef_))
        imp = imp / imp.sum() if imp.sum() else imp
        how = "absolute standardised coefficient"
    else:
        return pd.DataFrame()
    df = pd.DataFrame({"Feature": features, "Importance": imp}) \
           .sort_values("Importance", ascending=False).reset_index(drop=True)
    df["Kind"] = how
    return df


def save_artifacts(res) -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = {}
    with open(os.path.join(MODELS_DIR, "covid_cases_model.pkl"), "wb") as f:
        pickle.dump(res["best_model"], f)
    paths["model"] = os.path.join(MODELS_DIR, "covid_cases_model.pkl")

    with open(os.path.join(MODELS_DIR, "model_features.pkl"), "wb") as f:
        pickle.dump(res["features"], f)
    paths["features"] = os.path.join(MODELS_DIR, "model_features.pkl")

    meta = {
        "best_model": res["best_name"],
        "features": res["features"],
        "target": TARGET,
        "test_days": TEST_DAYS,
        "split": "chronological (final 21 rows held out)",
        "tuning": "GridSearchCV with TimeSeriesSplit(4)",
        "results": res["results"],
        "best_params": res["best_params"],
        "cleaning_report": res["report"].as_dict(),
        "leakage_note": ("All predictors are lagged by at least one day, so no "
                         "same-row information reaches the target."),
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    paths["metadata"] = os.path.join(MODELS_DIR, "model_metadata.json")

    imp = feature_importance(res["best_model"], res["features"])
    if len(imp):
        imp.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index=False)
        paths["importance"] = os.path.join(MODELS_DIR, "feature_importance.csv")
    return paths


if __name__ == "__main__":
    res = train_and_evaluate()
    paths = save_artifacts(res)
    print("\nSaved:")
    for k, v in paths.items():
        print("  %-10s %s" % (k, v))
    imp = feature_importance(res["best_model"], res["features"])
    print("\nTop features:")
    print(imp.head(10).to_string(index=False))

