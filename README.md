# 🦠 COVID-19 Pakistan — End-to-End Data Science Project

> **This is a historical analysis, not a live tracker.**
> Every number here describes Pakistan between **26 February and 3 June 2020**. It does not
> reflect the current COVID-19 situation in Pakistan or anywhere else, and the forecasts are
> back-tests against a closed historical window.

Exploratory analysis, statistical testing, leakage-free machine learning, ARIMA forecasting,
a Streamlit dashboard and automated reports — all built on one cleaned dataset of
Pakistan's COVID-19 records from **26 Feb 2020 to 03 Jun 2020**.

<p align="left">
  <a href="https://covid19-pakistan-analysis-n43mjecpftd2atum26wylp.streamlit.app"><img alt="Open in Streamlit" src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="statsmodels" src="https://img.shields.io/badge/statsmodels-0.14-3F4F75">
  <img alt="Plotly" src="https://img.shields.io/badge/Plotly-5.x-3F4F75?logo=plotly&logoColor=white">
</p>

> ### 🌐 Live dashboard
> **<https://covid19-pakistan-analysis-n43mjecpftd2atum26wylp.streamlit.app>**
>
> The app reads the cleaned CSVs and the ARIMA forecast that are committed to this repo, so it
> always matches the numbers in the report. If the link does not load, the deployment needs to be
> reconnected — see [Deploying the dashboard](#deploying-the-dashboard-streamlit-community-cloud).

---

## Table of contents

- [Live dashboard](#-live-dashboard)
- [Project overview](#project-overview)
- [Key findings](#key-findings)
- [Data quality and what changed](#data-quality-and-what-changed)
- [Machine learning](#machine-learning)
- [ARIMA forecasting](#arima-forecasting)
- [Project structure](#project-structure)
- [Tech stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations](#limitations)
- [Deploying the dashboard](#deploying-the-dashboard-streamlit-community-cloud)
- [Author](#author)

---

## Project overview

This project takes the raw Pakistan COVID-19 feed and carries it through a full analytical
pipeline:

| Stage | What it does |
|---|---|
| **Cleaning** | De-duplicates records, normalises 129 city spellings to 117, flags blank exposure sources |
| **EDA** | Daily / province / city / exposure-source analysis, 18 static figures, 6 interactive charts |
| **Statistics** | Descriptive profile, correlation, chi-square, ANOVA, normality and non-parametric tests |
| **Machine learning** | 4 regression learners with lagged features and a chronological split |
| **Forecasting** | ARIMA with an AIC order search and a 21-day hold-out |
| **Delivery** | Streamlit dashboard, Excel workbook, text insights report |

The whole pipeline lives in `src/` and is reproducible end to end — see [Usage](#usage).

---

## Key findings

| Metric | Value |
|---|---:|
| Date range | 26 Feb – 03 Jun 2020 |
| Cleaned records | **2,711** |
| Provinces | 8 |
| Cities (after normalisation) | **117** |
| Total cases | **83,985** |
| Total deaths | **1,728** |
| Total recovered | **24,753** |
| Case fatality rate | **2.06 %** |
| Recovery rate | **29.47 %** |
| Peak day | **4,677** cases on 03 Jun 2020 |

### Top provinces

| Rank | Province | Cases | CFR |
|:--:|---|---:|---:|
| 1 | Sindh | 32,858 | 1.66 % |
| 2 | Punjab | 31,096 | 1.91 % |
| 3 | Khyber Pakhtunkhwa | 10,259 | 4.61 % |
| 4 | Balochistan | 5,125 | 0.96 % |
| 5 | Islamabad Capital Territory | 3,523 | 1.02 % |

Sindh and Punjab together carry **76 %** of all cases.

### Top cities

| Rank | City | Province | Cases | CFR |
|:--:|---|---|---:|---:|
| 1 | Karachi | Sindh | **29,346** | 1.70 % |
| 2 | Lahore | Punjab | **28,015** | 2.08 % |
| 3 | Quetta | Balochistan | **5,109** | 0.96 % |
| 4 | Peshawar | Khyber Pakhtunkhwa | 4,923 | 5.38 % |
| 5 | Islamabad | Islamabad Capital Territory | 3,523 | 1.02 % |

Karachi and Lahore alone account for **68 %** of national cases.

### Exposure source

| Source | Cases | Share |
|---|---:|---:|
| Local – Social Contact | 58,918 | 70.15 % |
| Unknown | 22,659 | 26.98 % |
| Tableeghi Jamaat | 1,115 | 1.33 % |
| Iran / Taftan | 1,053 | 1.25 % |

> **Read "Unknown" carefully.** 36 source rows had a blank `Travel_history` and carry
> **19,200 cases (22.9 % of the total)**, concentrated in Punjab (16,520) and Balochistan
> (2,680). They are labelled `Unknown` so the category is a **reporting gap**, not a
> transmission route. The remaining ~3,459 `Unknown` cases were explicitly recorded that way.

---

## Data quality and what changed

The earlier version of this project shipped with several data problems. All of them were
fixed, and the pipeline now documents them:

| Issue | Before | After |
|---|---|---|
| Exact duplicate rows | 2 left in | removed (2,798 → 2,796) |
| City spellings | `Tor Ghar` / `Torghar` / `Tor  Ghar`, `Mirpur Khas` / `Mirpurkhas`, `Jehlum`, `Kurrum`, `Tando Alahyar` all counted separately → **125–132 "cities"** | normalised to **117 real cities** |
| Duplicate records on the analytical key | 85 groups summing different rows | collapsed by summing `Cases` / `Deaths` / `Recovered` → **2,711 rows** |
| Blank `Travel_history` | silently filled with `Unknown`, not mentioned anywhere | flagged in the data (`Unknown`) **and explained** here and in the reports |
| Missing calendar days | dropped, so the daily series skipped 8 dates | reindexed to a complete 99-day series with zeros |
| README totals | 83,986 / 1,728 / 24,754, 125 cities | 83,985 / 1,728 / 24,753, 117 cities |
| README top cities | Lahore "~15,000", 3rd "Peshawar ~4,000" | Lahore **28,015**, 3rd is **Quetta 5,109** |

The corrected totals are asserted in `src/data_pipeline.py`, so any future change to the
cleaning rules shows up immediately in the reports.

---

## Machine learning

> **This section was rebuilt.** The previous model predicted daily `Cases` using the
> **same row's** `Deaths` and `Recovered` and a **random** train/test split, which reported
> an R² of 0.75 that meant nothing. It also imported `GridSearchCV` without ever calling it.

### What the model does now

- **Only lagged predictors.** Every feature comes from day *t-1* or earlier: lag-1/3/7 case
  counts, lag-1 deaths and recoveries, 7- and 14-day rolling mean/std/max, cumulative cases
  and the growth rate up to *t-1*. No same-day information reaches the target.
- **Chronological split.** The final 21 days are held out, so the test window sits strictly
  after the training window.
- **Real tuning.** Random Forest and Gradient Boosting run through `GridSearchCV` with
  `TimeSeriesSplit(4)` inside the training window. Best parameters are saved to
  `models/model_metadata.json`.
- **Honest benchmark.** A naive "tomorrow = today" model is trained and scored alongside,
  because a forecast is only useful if it beats that.

### Results (21-day hold-out)

| Model | R² | RMSE | MAE | MAPE |
|---|---:|---:|---:|---:|
| **Naive (tomorrow = today)** | **0.779** | **434** | **347** | **15.3 %** |
| Gradient Boosting (tuned) | −0.264 | 1,039 | 720 | 26.3 % |
| Random Forest (tuned) | −0.358 | 1,077 | 719 | 25.0 % |
| Ridge (tuned) | −0.500 | 1,131 | 884 | 34.4 % |
| Linear Regression | −12.065 | 3,339 | 2,739 | 124 % |

**The honest conclusion:** with 62 training days on a steeply rising curve, none of the
learners beats the naive benchmark. Reporting that is more useful than quoting an inflated
R². The pipeline demonstrates correct methodology — lagged features, time-ordered
validation, real tuning and a benchmark — and it is the reason the forecasting job is left
to ARIMA rather than to a general-purpose regressor.

Feature importances (best learner) are saved to `models/feature_importance.csv`; lag-7
cases, lag-1 CFR and the 14-day rolling mean dominate.

---

## ARIMA forecasting

The daily series is rebuilt from the cleaned pipeline (99 calendar days, gap days as zero),
so the model sees an even time index.

- **Order selection:** AIC grid search over `p ∈ {0,1,2,3,7}`, `d ∈ {0,1,2}`, `q ∈ {0,1,2,3}`
- **Selected:** ARIMA(7, 2, 1)
- **Validation:** refit on the first 78 days, forecast the final 21

| Metric | ARIMA | Naive | Seasonal naive (t-7) |
|---|---:|---:|---:|
| Hold-out MAE | 644.5 | **347.2** | 860.8 |
| Hold-out RMSE | 768.6 | — | — |
| Hold-out MAPE | 32.6 % | — | — |

ARIMA does **not** beat the naive one-day-ahead model, but it does beat the seasonal naive.
The 30-day projection is therefore presented as a **scenario envelope**, not a point
prediction:

| 30-day projection | Value |
|---|---:|
| Total forecast cases | **247,996** |
| Mean per day | 8,267 |
| Peak day | 11,553 on 03 Jul 2020 |
| Minimum day | 4,842 |

Full outputs: `outputs/arima_forecast_30d.csv`, `outputs/arima_holdout.csv`,
`outputs/arima_order_grid.csv`, `outputs/arima_summary.json`.

---

## Project structure

```text
covid-19-portfolio/
│
├── app/
│   └── app.py                     # Streamlit dashboard
│
├── data/
│   ├── raw/
│   │   └── PK COVID-19-3jun.csv
│   └── processed/
│       ├── PK_COVID_19_cleaned.csv   # 2,711 cleaned records
│       ├── daily_features.csv        # 99-day daily series + lag-free features
│       ├── province_summary.csv
│       ├── city_summary.csv
│       ├── travel_summary.csv
│       └── model_table.csv           # supervised table with lagged predictors
│
├── src/
│   ├── data_pipeline.py           # cleaning, normalisation, aggregates
│   ├── ml_models.py               # lagged features, time split, GridSearchCV
│   ├── arima_forecast.py          # order search, hold-out, 30-day forecast
│   ├── visualizations.py          # 18 static figures + 6 interactive charts
│   └── reports.py                 # Excel workbook + text insights
│
├── models/
│   ├── covid_cases_model.pkl
│   ├── model_features.pkl
│   ├── model_metadata.json        # metrics, best params, cleaning report
│   ├── feature_importance.csv
│   └── arima_model.pkl
│
├── notebooks/
│   └── COVID_19_analysis.ipynb    # narrative walkthrough of the same pipeline
│
├── outputs/
│   ├── figures/                   # 18 PNG charts
│   ├── interactive/               # 6 Plotly HTML charts (~10 KB each, CDN mode)
│   ├── COVID19_Analysis_Report.xlsx
│   ├── COVID19_Insights_Report.txt
│   └── arima_*.csv / .json
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech stack

| Technology | Purpose |
|---|---|
| Python 3.14 | Core language |
| pandas / NumPy | Data wrangling and numerics |
| Matplotlib / Seaborn | Static visualisations |
| Plotly | Interactive HTML charts |
| scikit-learn | Regression learners, `GridSearchCV`, `TimeSeriesSplit` |
| statsmodels | ARIMA, ADF stationarity tests |
| SciPy | Statistical tests |
| Streamlit | Interactive dashboard |
| openpyxl | Excel report |

---

## Installation

**Prerequisites:** Python 3.10 or newer and `pip`.

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**Dashboard only** (this is what Streamlit Cloud installs):

```bash
pip install -r requirements.txt
```

**Full pipeline** (adds matplotlib, seaborn, scipy and Jupyter so the notebooks and the
figure/report scripts run):

```bash
pip install -r requirements-dev.txt
```

---

## Usage

### 1. Rebuild the data

```bash
python src/data_pipeline.py
```

Writes the cleaned CSV plus every aggregate and the supervised model table.

### 2. Train the models

```bash
python src/ml_models.py
```

Runs the tuning grid, prints the hold-out comparison against the naive benchmark, and saves
the artefacts and the metadata JSON.

### 3. Fit ARIMA and build the forecast

```bash
python src/arima_forecast.py
```

### 4. Regenerate every chart

```bash
python src/visualizations.py
```

18 PNGs into `outputs/figures`, 6 interactive HTML files into `outputs/interactive`.

### 5. Rebuild the reports

```bash
python src/reports.py
```

### 6. Launch the dashboard

```bash
streamlit run app/app.py
```

Open <http://localhost:8501>.

### 7. Run everything at once

```bash
python src/data_pipeline.py && python src/ml_models.py && \
python src/arima_forecast.py && python src/visualizations.py && python src/reports.py
```

---

## Limitations

- **The ML models do not beat the naive benchmark.** That is reported, not hidden. The
  earlier 0.75 R² came from same-day leakage and a random split.
- **ARIMA does not beat the naive one-day-ahead model** either, though it does beat the
  seasonal naive. The projection should be read as a scenario band.
- **`Recovered` is under-reported**, so the recovery rate is a lower bound.
- **`Unknown` exposure source is a reporting gap**, and its 22,659 cases inflate the
  apparent local-transmission share.
- **Exposure source and province are correlated**, so provincial CFR differences are not a
  clean measure of care quality.
- **Only 91 distinct dates over a 99-day window**, and only 62 usable training rows after
  lag construction. Any conclusion here is descriptive, not predictive at scale.
- **Active cases are an estimate,** computed as `Cases − Deaths − Recovered`. The source
  reports recoveries in batches, so **402 of the 2,711 records (14.8%)** would show a negative
  row-level active count — Recovered exceeds Cases on those rows because recoveries were
  reported on days with no new cases for that city. The figure is never clamped to zero,
  because clamping would hide the inconsistency. The national aggregate (57,504) is positive
  and usable, but it is an approximation, not a verified active-case count.
- **The historical framing matters.** Present this as a retrospective study of Feb–Jun 2020,
  not as a report on the present. The ARIMA projection covers 04 Jun – 03 Jul 2020 and has
  already been overtaken by events.
- **The dataset is a convenience sample**, not an official government feed, so totals may
  differ from Ministry of Health figures for the same dates.

---

## Deploying the dashboard (Streamlit Community Cloud)

### "The app's code is not connected to a remote GitHub repository"

That message comes from Streamlit Cloud, not from this project. It means Streamlit cannot read
the repository it was pointed at. There are only four real causes:

| Cause | Fix |
|---|---|
| **The repository is private** | Free Streamlit Cloud can only read **public** repos. Open the repo on GitHub → **Settings → General → Danger Zone → Change visibility → Make public**. |
| **Streamlit was never granted access to GitHub** | In Streamlit Cloud click your avatar → **Settings → Linked accounts** → reconnect GitHub, and when GitHub asks, choose **"All repositories"** *(or explicitly tick this repo if you use "Only select repositories")*. |
| **The app was created before the code was pushed** | Delete the broken app in Streamlit Cloud (**⋮ → Delete**) and create it again after the repo is public. |
| **Wrong branch or wrong file path** | Use `main` and `app/app.py` exactly — see the settings table below. |

The repository for this project is already public-code-ready: it contains `requirements.txt`
at the root, the dashboard at `app/app.py`, and the data and model artefacts the app reads.

The live app is expected at
**<https://covid19-pakistan-analysis-n43mjecpftd2atum26wylp.streamlit.app>**.
If that URL is dead, redeploying over the same app keeps the URL; creating a brand-new app
produces a different one, so update this README when that happens.

### Exact deployment settings

When you click **New app** in Streamlit Cloud, fill the form exactly like this:

| Field | Value |
|---|---|
| Repository | `Bilaldataorbit/covid19-pakistan-analysis` |
| Branch | `main` |
| Main file path | `app/app.py` |
| App URL | anything you like, e.g. `covid19-pakistan` |

Optional but recommended: open **Advanced settings** and set **Python version = 3.12**.

### What the app needs at runtime

The dashboard reads only files that are committed to the repo, so no database or API keys are
required:

```text
data/processed/PK_COVID_19_cleaned.csv
data/processed/daily_features.csv
data/processed/province_summary.csv
data/processed/city_summary.csv
data/processed/travel_summary.csv
models/covid_cases_model.pkl        # optional - see the note below
models/model_features.pkl
models/model_metadata.json
outputs/arima_forecast_30d.csv
outputs/arima_summary.json
```

### The app survives a model it cannot load

A pickle is tied to the library versions that wrote it. Streamlit Cloud installs whatever
`requirements.txt` allows, which may be a different scikit-learn build than the one used to
train the model. The dashboard therefore treats the trained model as **optional**:

- if the pickle loads → the "Next-day scenario" panel shows naive, 7-day average **and** the learner
- if the pickle does not load → the panel still works with the two benchmark estimates, a
  caption explains why, and every other tab (trend, geography, exposure source, distribution,
  ARIMA forecast) is unaffected

This was verified by deliberately corrupting the pickle and re-importing the app: it still
starts, still reads the metadata and still draws the ARIMA forecast.

### If the deploy build fails

| Log message | Meaning | Fix |
|---|---|---|
| `ModuleNotFoundError: streamlit` | wrong requirements file | `requirements.txt` must sit in the **repo root** |
| `FileNotFoundError: data/processed/...` | data not committed | re-add the CSV files and push |
| `InconsistentVersionWarning` on unpickle | scikit-learn mismatch | harmless, the app falls back automatically |
| `No such file: app/app.py` | wrong main file path | use `app/app.py`, not `app.py` |

### Deploy from the command line instead

```bash
git add -A
git commit -m "deploy: dashboard ready for Streamlit Cloud"
git push origin main
```

Then create the app in the Streamlit Cloud web UI with the settings table above.

## Author

**Bilal Raza**

- GitHub: [@Bilaldataorbit](https://github.com/Bilaldataorbit)
- Email: dataorbit.official@gmail.com

## License

**MIT** — see [LICENSE](LICENSE) for the full text.

```text
MIT License · Copyright (c) 2026 Bilal Raza
```

The licence covers the code and the analysis. The underlying observations remain subject to
the terms of their original source.

## Data limitations

Short version, so nobody has to hunt for it:

- **Historical window only.** 26 February – 3 June 2020. Not a live COVID-19 tracker.
- **Missing calendar dates are filled with zero** in the daily series. That is an analytical
  assumption (no record that day), not proof that no cases occurred.
- **Active cases are estimated** as `Cases − Deaths − Recovered`. 402 records (14.8%) would show
  a negative row-level value because recoveries were reported in batches; the values are left
  as computed rather than clamped so the inconsistency stays visible.
- **`Unknown` exposure source is a reporting gap** — 19,200 of its 22,659 cases come from
  36 rows whose source field was blank.
- **Forecasts are exploratory.** The ML learners do not beat the naive baseline and ARIMA does
  not beat it either; both results are reported rather than hidden.
