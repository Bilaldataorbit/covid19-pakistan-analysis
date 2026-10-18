"""
src/data_pipeline.py
====================
Single source of truth for loading, cleaning and engineering the
Pakistan COVID-19 dataset (26 Feb 2020 - 03 Jun 2020).

Everything downstream (notebook, models, app, reports) imports from here,
so the cleaning rules are written once and applied consistently.

Cleaning steps
--------------
1. strip whitespace from every text column
2. drop the exact duplicate rows in the raw feed
3. normalise city and province spellings to one canonical form
4. flag rows whose Travel_history was blank before filling it
5. collapse occasional (Date, Province, City, Travel) duplicates by summing
6. build the daily / province / city / travel aggregates
7. engineer the model-ready daily table with lag features (no leakage)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "data", "raw", "PK COVID-19-3jun.csv")
CLEAN_PATH = os.path.join(ROOT, "data", "processed", "PK_COVID_19_cleaned.csv")
DAILY_PATH = os.path.join(ROOT, "data", "processed", "daily_features.csv")
AGG_DIR = os.path.join(ROOT, "data", "processed")

# ------------------------------------------------------- canonical spellings
CITY_FIXES = {
    "tor ghar": "Tor Ghar",
    "torghar": "Tor Ghar",
    "tor  ghar": "Tor Ghar",
    "mirpur khas": "Mirpur Khas",
    "mirpurkhas": "Mirpur Khas",
    "mirpur": "Mirpur",
    "jehlum": "Jhelum",
    "jhelum": "Jhelum",
    "kurrum": "Kurram",
    "kurram": "Kurram",
    "tando alahyar": "Tando Allahyar",
    "tando allahyar": "Tando Allahyar",
    "chitral lower": "Chitral",
    "chitral upper": "Chitral",
    "chitral": "Chitral",
    "kohistan upper": "Kohistan",
    "kohistan lower": "Kohistan",
    "kohistan": "Kohistan",
    "islamabad": "Islamabad",
    "gujrat": "Gujrat",
    "hangu": "Hangu",
    "mardan": "Mardan",
    "rahim yar khan": "Rahim Yar Khan",
    "dera ismail khan": "Dera Ismail Khan",
    "dera ghazi khan": "Dera Ghazi Khan",
    "rawalpindi": "Rawalpindi",
    "muzaffarabad": "Muzaffarabad",
}

PROVINCE_FIXES = {
    "azad jummu kashmir": "Azad Jammu Kashmir",
    "azad jammu kashmir": "Azad Jammu Kashmir",
    "azad kashmir": "Azad Jammu Kashmir",
    "federal administration tribal area": "Federally Administered Tribal Areas",
    "federally administered tribal areas": "Federally Administered Tribal Areas",
    "fata": "Federally Administered Tribal Areas",
    "gilgit baltistan": "Gilgit-Baltistan",
    "gilgit-baltistan": "Gilgit-Baltistan",
    "islamabad capital territory": "Islamabad Capital Territory",
    "ict": "Islamabad Capital Territory",
    "khyber pakhtunkhwa": "Khyber Pakhtunkhwa",
    "kpk": "Khyber Pakhtunkhwa",
    "baluchistan": "Balochistan",
    "balochistan": "Balochistan",
    "sindh": "Sindh",
    "punjab": "Punjab",
}

TRAVEL_FIXES = {
    "local - social contact": "Local - Social Contact",
    "local - covid relative": "Local - Covid Relative",
    "local transmission": "Local - Social Contact",
    "iran/taftan": "Iran / Taftan",
    "iran": "Iran / Taftan",
    "taftan": "Iran / Taftan",
    "tableeghi jamaat": "Tableeghi Jamaat",
    "international passenger": "International Passenger",
    "unknown": "Unknown",
    "nan": "Unknown",
}


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


# --------------------------------------------------------------------- loader
@dataclass
class CleanReport:
    raw_rows: int
    after_exact_dupes: int
    after_city_norm_groups: int
    final_rows: int
    cities_raw: int
    cities_final: int
    provinces_raw: int
    provinces_final: int
    missing_travel_rows: int
    missing_travel_cases: int
    total_cases: int
    total_deaths: int
    total_recovered: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include=["object", "str"]).columns:
        df[c] = df[c].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ("Cases", "Deaths", "Recovered"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def clean(df: pd.DataFrame | None = None, verbose: bool = True) -> tuple[pd.DataFrame, CleanReport]:
    """Apply every cleaning rule and return the tidy frame plus a report."""
    if df is None:
        df = load_raw()

    raw_rows = len(df)
    cities_raw = df["City"].nunique()
    provinces_raw = df["Province"].nunique()

    # 1. exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    after_exact = len(df)

    # 2. normalise the categorical spellings
    df["City"] = df["City"].map(lambda x: CITY_FIXES.get(_norm_key(x), str(x).strip()))
    df["Province"] = df["Province"].map(lambda x: PROVINCE_FIXES.get(_norm_key(x), str(x).strip()))

    # 3. missing travel history -> Unknown, but remember it was missing
    df["Travel_Missing"] = df["Travel_history"].isna() | (
        df["Travel_history"].astype(str).str.strip().str.lower().isin(["", "nan", "none"])
    )
    missing_rows = int(df["Travel_Missing"].sum())
    missing_cases = int(df.loc[df["Travel_Missing"], "Cases"].sum())
    df["Travel_history"] = (
        df["Travel_history"]
        .fillna("Unknown")
        .map(lambda x: TRAVEL_FIXES.get(_norm_key(x), str(x).strip()))
    )

    # 4. collapse remaining duplicates on the analytical key
    key = ["Date", "Province", "City", "Travel_history"]
    agg = {"Cases": "sum", "Deaths": "sum", "Recovered": "sum", "Travel_Missing": "max"}
    df = df.groupby(key, as_index=False).agg(agg)
    df = df.sort_values(["Date", "Province", "City"]).reset_index(drop=True)

    cities_final = df["City"].nunique()
    provinces_final = df["Province"].nunique()

    rep = CleanReport(
        raw_rows=raw_rows,
        after_exact_dupes=after_exact,
        after_city_norm_groups=len(df),
        final_rows=len(df),
        cities_raw=cities_raw,
        cities_final=cities_final,
        provinces_raw=provinces_raw,
        provinces_final=provinces_final,
        missing_travel_rows=missing_rows,
        missing_travel_cases=missing_cases,
        total_cases=int(df["Cases"].sum()),
        total_deaths=int(df["Deaths"].sum()),
        total_recovered=int(df["Recovered"].sum()),
    )
    if verbose:
        print("Cleaning report")
        print("  raw rows                :", rep.raw_rows)
        print("  after dropping duplicates:", rep.after_exact_dupes, "(-%d)" % (raw_rows - rep.after_exact_dupes))
        print("  final rows              :", rep.final_rows)
        print("  cities  %d -> %d        :" % (rep.cities_raw, rep.cities_final))
        print("  provinces %d -> %d      :" % (rep.provinces_raw, rep.provinces_final))
        print("  missing travel rows     :", rep.missing_travel_rows,
              "(%d cases, %.2f%% of all)" % (rep.missing_travel_cases,
                                              100 * rep.missing_travel_cases / rep.total_cases))
        print("  totals cases/deaths/rec : %d / %d / %d"
              % (rep.total_cases, rep.total_deaths, rep.total_recovered))
    return df, rep


# ----------------------------------------------------------------- aggregates
def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b-%Y")
    df["Month_Short"] = df["Date"].dt.strftime("%b")
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.day_name()
    df["DayOfWeek_Num"] = df["Date"].dt.dayofweek
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["Active"] = df["Cases"] - df["Deaths"] - df["Recovered"]
    df["CFR"] = np.where(df["Cases"] > 0, df["Deaths"] / df["Cases"] * 100, 0.0)
    df["Recovery_Rate"] = np.where(df["Cases"] > 0, df["Recovered"] / df["Cases"] * 100, 0.0)
    df["Is_Local"] = df["Travel_history"].str.contains("Local", case=False, na=False).astype(int)
    df["Is_Tableeghi"] = df["Travel_history"].str.contains("Tableeghi", case=False, na=False).astype(int)
    df["Is_Imported"] = (~df["Travel_history"].str.contains("Local|Unknown", case=False, na=False)).astype(int)
    return df


def daily_table(df: pd.DataFrame) -> pd.DataFrame:
    d = (df.groupby("Date")
           .agg(Cases=("Cases", "sum"), Deaths=("Deaths", "sum"), Recovered=("Recovered", "sum"))
           .reset_index()
           .sort_values("Date"))
    d = d.asfreq("D") if False else d.set_index("Date").asfreq("D").reset_index()
    d[["Cases", "Deaths", "Recovered"]] = d[["Cases", "Deaths", "Recovered"]].fillna(0).astype(int)
    d["Cumulative_Cases"] = d["Cases"].cumsum()
    d["Cumulative_Deaths"] = d["Deaths"].cumsum()
    d["Cumulative_Recovered"] = d["Recovered"].cumsum()
    d["Active"] = d["Cumulative_Cases"] - d["Cumulative_Deaths"] - d["Cumulative_Recovered"]
    d["CFR"] = (d["Cumulative_Deaths"] / d["Cumulative_Cases"] * 100).round(3)
    d["Recovery_Rate"] = (d["Cumulative_Recovered"] / d["Cumulative_Cases"] * 100).round(3)
    d["MA7_Cases"] = d["Cases"].rolling(7, min_periods=1).mean().round(2)
    d["MA7_Deaths"] = d["Deaths"].rolling(7, min_periods=1).mean().round(2)
    d["Growth_Rate"] = (d["Cases"].pct_change() * 100).replace([np.inf, -np.inf], np.nan).round(2)
    d["Day_Num"] = np.arange(1, len(d) + 1)
    return d


def province_table(df: pd.DataFrame) -> pd.DataFrame:
    p = (df.groupby("Province")
           .agg(Cases=("Cases", "sum"), Deaths=("Deaths", "sum"),
                Recovered=("Recovered", "sum"), Records=("Cases", "size"))
           .reset_index())
    p["Active"] = p["Cases"] - p["Deaths"] - p["Recovered"]
    p["CFR"] = (p["Deaths"] / p["Cases"] * 100).round(2)
    p["Recovery_Rate"] = (p["Recovered"] / p["Cases"] * 100).round(2)
    p["Case_Share"] = (p["Cases"] / p["Cases"].sum() * 100).round(2)
    top_city = (df.groupby(["Province", "City"])["Cases"].sum().reset_index()
                  .sort_values(["Province", "Cases"], ascending=[True, False])
                  .drop_duplicates("Province").set_index("Province")["City"])
    p["Top_City"] = p["Province"].map(top_city)
    return p.sort_values("Cases", ascending=False).reset_index(drop=True)


def city_table(df: pd.DataFrame) -> pd.DataFrame:
    c = (df.groupby(["City", "Province"])
           .agg(Cases=("Cases", "sum"), Deaths=("Deaths", "sum"),
                Recovered=("Recovered", "sum"), Records=("Cases", "size"))
           .reset_index())
    c["Active"] = c["Cases"] - c["Deaths"] - c["Recovered"]
    c["CFR"] = (c["Deaths"] / c["Cases"] * 100).round(2)
    c["Recovery_Rate"] = (c["Recovered"] / c["Cases"] * 100).round(2)
    return c.sort_values("Cases", ascending=False).reset_index(drop=True)


def travel_table(df: pd.DataFrame) -> pd.DataFrame:
    t = (df.groupby("Travel_history")
           .agg(Cases=("Cases", "sum"), Deaths=("Deaths", "sum"),
                Recovered=("Recovered", "sum"), Records=("Cases", "size"),
                Missing_Flag=("Travel_Missing", "sum"))
           .reset_index())
    t["CFR"] = (t["Deaths"] / t["Cases"] * 100).round(2)
    t["Case_Share"] = (t["Cases"] / t["Cases"].sum() * 100).round(2)
    return t.sort_values("Cases", ascending=False).reset_index(drop=True)


# ------------------------------------------------- leakage-free model table
def model_table(daily: pd.DataFrame, horizon_features: int = 7) -> pd.DataFrame:
    """Build a supervised learning table for forecasting daily cases.

    Every predictor for day t is taken from day t-1 or earlier, so the frame can be
    used with a time-ordered split and there is no same-row leakage.
    """
    m = daily.copy().sort_values("Date").reset_index(drop=True)

    m["Lag_Cases_1"] = m["Cases"].shift(1)
    m["Lag_Cases_3"] = m["Cases"].shift(3)
    m["Lag_Cases_7"] = m["Cases"].shift(7)
    m["Lag_Deaths_1"] = m["Deaths"].shift(1)
    m["Lag_Recovered_1"] = m["Recovered"].shift(1)
    m["Roll_Mean_7"] = m["Cases"].shift(1).rolling(7).mean()
    m["Roll_Mean_14"] = m["Cases"].shift(1).rolling(14).mean()
    m["Roll_Std_7"] = m["Cases"].shift(1).rolling(7).std()
    m["Roll_Max_14"] = m["Cases"].shift(1).rolling(14).max()
    m["Cumul_Cases_Lag"] = m["Cases"].shift(1).cumsum()
    m["DayOfWeek_Num"] = m["Date"].dt.dayofweek
    m["Month"] = m["Date"].dt.month
    m["Day_Num"] = np.arange(1, len(m) + 1)
    m["CFR_Lag_1"] = (m["Deaths"].shift(1).cumsum() / m["Cases"].shift(1).cumsum() * 100).round(3)
    m["Growth_Lag_1"] = m["Growth_Rate"].shift(1)

    m = m.dropna().reset_index(drop=True)
    return m


MODEL_FEATURES = [
    "Lag_Cases_1", "Lag_Cases_3", "Lag_Cases_7",
    "Lag_Deaths_1", "Lag_Recovered_1",
    "Roll_Mean_7", "Roll_Mean_14", "Roll_Std_7", "Roll_Max_14",
    "Cumul_Cases_Lag", "DayOfWeek_Num", "Month", "CFR_Lag_1", "Growth_Lag_1",
]
TARGET = "Cases"


# ------------------------------------------------------------------ entrypoint
def build_all(verbose: bool = True):
    raw = load_raw()
    df, rep = clean(raw, verbose=verbose)
    df = add_calendar(df)
    daily = daily_table(df)
    prov = province_table(df)
    city = city_table(df)
    travel = travel_table(df)
    model = model_table(daily)

    os.makedirs(AGG_DIR, exist_ok=True)
    df.drop(columns=["Travel_Missing"]).to_csv(CLEAN_PATH, index=False)
    daily.to_csv(DAILY_PATH, index=False)
    prov.to_csv(os.path.join(AGG_DIR, "province_summary.csv"), index=False)
    city.to_csv(os.path.join(AGG_DIR, "city_summary.csv"), index=False)
    travel.to_csv(os.path.join(AGG_DIR, "travel_summary.csv"), index=False)
    model.to_csv(os.path.join(AGG_DIR, "model_table.csv"), index=False)

    if verbose:
        print("\nWrote:")
        for p in [CLEAN_PATH, DAILY_PATH]:
            print("  ", p)
    return dict(df=df, daily=daily, province=prov, city=city, travel=travel,
                model=model, report=rep)


if __name__ == "__main__":
    out = build_all()
    print("\nDaily rows:", len(out["daily"]))
    print("Model rows:", len(out["model"]), "| features:", len(MODEL_FEATURES))
    print(out["daily"].head(3).to_string())

