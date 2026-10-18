"""
app/app.py
==========
Streamlit dashboard for the COVID-19 Pakistan analysis.

The dashboard reads the cleaned data and the trained model from the project folders,
so it always shows the same numbers as the notebook and the reports.

Run locally with:  streamlit run app/app.py
"""
from __future__ import annotations

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")

st.set_page_config(page_title="COVID-19 Pakistan | Dashboard",
                   page_icon="🦠", layout="wide",
                   initial_sidebar_state="expanded")

ACCENT = "#2E75B6"
WARN = "#BF8F00"
GOOD = "#375623"


# --------------------------------------------------------------------- loaders
@st.cache_data
def load_frames():
    clean = pd.read_csv(os.path.join(DATA, "PK_COVID_19_cleaned.csv"), parse_dates=["Date"])
    daily = pd.read_csv(os.path.join(DATA, "daily_features.csv"), parse_dates=["Date"])
    prov = pd.read_csv(os.path.join(DATA, "province_summary.csv"))
    city = pd.read_csv(os.path.join(DATA, "city_summary.csv"))
    travel = pd.read_csv(os.path.join(DATA, "travel_summary.csv"))
    return clean, daily, prov, city, travel


@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODELS, "covid_cases_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS, "model_features.pkl"), "rb") as f:
        features = pickle.load(f)
    meta = {}
    meta_path = os.path.join(MODELS, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    forecast = None
    fc_path = os.path.join(BASE, "outputs", "arima_forecast_30d.csv")
    if os.path.exists(fc_path):
        forecast = pd.read_csv(fc_path, parse_dates=["Date"])
    return model, features, meta, forecast


clean, daily, prov, city, travel = load_frames()
model, features, meta, arima_fc = load_artifacts()

# ----------------------------------------------------------------------- header
st.title("🦠 COVID-19 Pakistan — interactive analysis")
st.caption(
    f"{len(clean):,} cleaned records · {clean['Date'].min():%d %b %Y} – "
    f"{clean['Date'].max():%d %b %Y} · {clean['Province'].nunique()} provinces · "
    f"{clean['City'].nunique()} cities"
)

# ----------------------------------------------------------------------- sidebar
st.sidebar.header("Filters")
dmin, dmax = clean["Date"].min().date(), clean["Date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(dmin, dmax),
                                   min_value=dmin, max_value=dmax)
province_sel = st.sidebar.multiselect("Province", sorted(clean["Province"].unique()))
travel_sel = st.sidebar.multiselect("Exposure source", sorted(clean["Travel_history"].unique()))

mask = (clean["Date"].dt.date >= date_range[0]) & (clean["Date"].dt.date <= date_range[1])
view = clean[mask]
if province_sel:
    view = view[view["Province"].isin(province_sel)]
if travel_sel:
    view = view[view["Travel_history"].isin(travel_sel)]

# -------------------------------------------------------------------------- KPIs
c1, c2, c3, c4, c5 = st.columns(5)
tc, td = int(view["Cases"].sum()), int(view["Deaths"].sum())
tr = int(view["Recovered"].sum())
c1.metric("Total cases", f"{tc:,}")
c2.metric("Deaths", f"{td:,}")
c3.metric("Recovered", f"{tr:,}")
c4.metric("Case fatality", f"{(td / tc * 100 if tc else 0):.2f}%")
c5.metric("Active", f"{tc - td - tr:,}")

st.divider()

# ---------------------------------------------------------------------- tabs
tab_trend, tab_geo, tab_source, tab_dist, tab_model, tab_data = st.tabs(
    ["📈 Trend", "🗺️ Geography", "🦠 Exposure source", "📊 Distribution", "🤖 Model", "📋 Data"]
)

# ------------------------------------------------------------------ trend tab
with tab_trend:
    agg = (view.groupby("Date")
               .agg(Cases=("Cases", "sum"), Deaths=("Deaths", "sum"),
                    Recovered=("Recovered", "sum"))
               .reset_index().sort_values("Date"))
    agg["MA7"] = agg["Cases"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Date"], y=agg["Cases"], name="daily cases",
                         marker_color=ACCENT, opacity=.55))
    fig.add_trace(go.Scatter(x=agg["Date"], y=agg["MA7"], name="7-day average",
                             line=dict(color=WARN, width=3)))
    fig.update_layout(title="Daily cases with 7-day moving average",
                      template="plotly_white", height=430,
                      legend=dict(orientation="h", y=1.02), margin=dict(t=60))
    st.plotly_chart(fig, use_container_width=True)

    l, r = st.columns(2)
    with l:
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=agg["Date"], y=agg["Deaths"], name="deaths",
                                fill="tozeroy", line=dict(color=WARN)))
        f2.update_layout(title="Daily deaths", template="plotly_white", height=330)
        st.plotly_chart(f2, use_container_width=True)
    with r:
        cum = agg.assign(Cum=agg["Cases"].cumsum())
        f3 = go.Figure(go.Scatter(x=cum["Date"], y=cum["Cum"], fill="tozeroy",
                                  line=dict(color=ACCENT), name="cumulative"))
        f3.update_layout(title="Cumulative cases", template="plotly_white", height=330)
        st.plotly_chart(f3, use_container_width=True)

# ------------------------------------------------------------------- geo tab
with tab_geo:
    p = (view.groupby("Province")
             .agg(Cases=("Cases", "sum"), Deaths=("Deaths", "sum"),
                  Recovered=("Recovered", "sum")).reset_index())
    p["CFR"] = np.where(p["Cases"] > 0, p["Deaths"] / p["Cases"] * 100, 0).round(2)
    c = (view.groupby(["City", "Province"])
             .agg(Cases=("Cases", "sum")).reset_index()
             .sort_values("Cases", ascending=False).head(20))

    l, r = st.columns(2)
    with l:
        f = px.bar(p.sort_values("Cases"), x="Cases", y="Province", orientation="h",
                   color="CFR", color_continuous_scale="YlOrRd",
                   title="Cases by province (colour = CFR)")
        f.update_layout(template="plotly_white", height=420)
        st.plotly_chart(f, use_container_width=True)
    with r:
        f = px.bar(c.sort_values("Cases"), x="Cases", y="City", orientation="h",
                   color="Province", title="Top 20 cities")
        f.update_layout(template="plotly_white", height=520)
        st.plotly_chart(f, use_container_width=True)

    f = px.treemap(view, path=["Province", "City"], values="Cases",
                   title="Case load by province and city")
    f.update_layout(template="plotly_white", height=520)
    st.plotly_chart(f, use_container_width=True)

# ---------------------------------------------------------------- source tab
with tab_source:
    t = (view.groupby("Travel_history")
             .agg(Cases=("Cases", "sum"), Deaths=("Deaths", "sum"),
                  Records=("Cases", "size")).reset_index())
    t["CFR"] = np.where(t["Cases"] > 0, t["Deaths"] / t["Cases"] * 100, 0).round(2)
    t = t.sort_values("Cases", ascending=False)

    l, r = st.columns(2)
    with l:
        f = px.pie(t.head(8), values="Cases", names="Travel_history",
                   title="Case share by exposure source", hole=.45)
        f.update_layout(template="plotly_white", height=420)
        st.plotly_chart(f, use_container_width=True)
    with r:
        f = px.bar(t, x="Cases", y="Travel_history", orientation="h", color="CFR",
                   color_continuous_scale="YlOrRd", title="Cases and CFR by source")
        f.update_layout(template="plotly_white", height=420)
        st.plotly_chart(f, use_container_width=True)

    st.info(
        "**Reading the 'Unknown' slice.** 36 source rows had a blank Travel_history. "
        f"They carry {int(clean.loc[clean['Travel_history'] == 'Unknown', 'Cases'].sum()):,} cases, "
        "so 'Unknown' is a reporting gap, not a transmission category."
    )

# -------------------------------------------------------------- distribution tab
with tab_dist:
    l, r = st.columns(2)
    with l:
        f = px.histogram(view, x="Cases", nbins=40, title="Distribution of case counts per record",
                         color_discrete_sequence=[ACCENT])
        f.update_layout(template="plotly_white", height=380, bargap=.05)
        st.plotly_chart(f, use_container_width=True)
    with r:
        f = px.box(view, x="Province", y="Cases", color="Province",
                   title="Record-level cases by province")
        f.update_layout(template="plotly_white", height=380, showlegend=False)
        f.update_xaxes(tickangle=35)
        st.plotly_chart(f, use_container_width=True)

    l, r = st.columns(2)
    with l:
        d = view.groupby("Date").agg(Cases=("Cases", "sum"),
                                     Deaths=("Deaths", "sum")).reset_index()
        f = px.histogram(d, x="Cases", nbins=25, title="Distribution of daily totals",
                         color_discrete_sequence=[WARN])
        f.update_layout(template="plotly_white", height=360, bargap=.05)
        st.plotly_chart(f, use_container_width=True)
    with r:
        f = px.violin(view, x="Is_Local" if "Is_Local" in view.columns else "Province",
                      y="Cases", title="Local transmission vs imported records" if "Is_Local" in view.columns
                      else "Cases by province", color="Province" if "Is_Local" not in view.columns else "Is_Local",
                      box=True)
        f.update_layout(template="plotly_white", height=360, showlegend=False)
        f.update_yaxes(type="log")
        st.plotly_chart(f, use_container_width=True)

    st.caption(f"{len(view):,} records in the current filter. "
               f"Median record = {view['Cases'].median():.0f} cases, "
               f"mean = {view['Cases'].mean():.1f} - the mean sits far above the median, "
               "which is why the distribution is plotted on a log axis.")

# ----------------------------------------------------------------- model tab
with tab_model:
    st.subheader("Forecasting models")
    if meta:
        st.caption(
            f"Best learner: **{meta.get('best_model')}** · split: {meta.get('split')} · "
            f"tuning: {meta.get('tuning')}"
        )
        res = pd.DataFrame(meta.get("results", {})).T.reset_index()
        res = res.rename(columns={"index": "model"})
        st.dataframe(res, use_container_width=True, hide_index=True)

        naive_rmse = meta["results"].get("Naive (tomorrow = today)", {}).get("RMSE")
        best_rmse = meta["results"].get(meta.get("best_model", ""), {}).get("RMSE")
        st.warning(
            "**Why there is no impressive R² here.** The predictors are all lagged and the split "
            "is chronological, so this score is genuine out-of-sample skill. On only 62 training "
            "days 'tomorrow = today' is a very strong benchmark"
            + (f" (RMSE {naive_rmse:,.0f} vs {best_rmse:,.0f} for the best learner)" if naive_rmse and best_rmse else "")
            + ", and the learners do not beat it. The earlier 0.75 R² came from using same-day "
              "deaths and recoveries plus a random split, which is leakage."
        )

    if arima_fc is not None:
        st.subheader("30-day ARIMA projection")
        hist = daily.tail(45)
        f = go.Figure()
        f.add_trace(go.Scatter(x=hist["Date"], y=hist["Cases"], name="observed",
                               line=dict(color=ACCENT)))
        f.add_trace(go.Scatter(x=arima_fc["Date"], y=arima_fc["Forecast"],
                               name="forecast", line=dict(color=WARN, width=3)))
        f.add_trace(go.Scatter(x=arima_fc["Date"], y=arima_fc["Upper_95"],
                               line=dict(width=0), showlegend=False))
        f.add_trace(go.Scatter(x=arima_fc["Date"], y=arima_fc["Lower_95"],
                               fill="tonexty", line=dict(width=0),
                               fillcolor="rgba(191,143,0,.18)", name="95% interval"))
        f.update_layout(template="plotly_white", height=430,
                        title="ARIMA forecast with a 95% interval",
                        legend=dict(orientation="h", y=1.02))
        st.plotly_chart(f, use_container_width=True)
        if os.path.exists(os.path.join(BASE, "outputs", "arima_summary.json")):
            with open(os.path.join(BASE, "outputs", "arima_summary.json"), encoding="utf-8") as fh:
                ar_sum = json.load(fh)
            st.caption(
                f"ARIMA{ar_sum['order']} · hold-out MAE {ar_sum['holdout_metrics']['MAE']:,.0f} "
                f"(naive {ar_sum['naive_holdout_mae']:,.0f}, "
                f"seasonal naive {ar_sum['seasonal_naive_holdout_mae']:,.0f}) · "
                f"30-day total {ar_sum['forecast_total']:,} cases. "
                "ARIMA beats the seasonal naive but not the one-day naive model, so read the "
                "projection as a scenario envelope."
            )

    # ------------------------------------------------------------ scenario tool
    st.subheader("Next-day scenario")
    st.write(
        "The learner does not beat the naive benchmark, so this tool deliberately shows three "
        "estimates side by side instead of a single confident number."
    )
    a, b, c = st.columns(3)
    lag1 = a.number_input("Cases yesterday", min_value=0, value=int(daily["Cases"].iloc[-1]))
    lag3 = b.number_input("Cases 3 days ago", min_value=0, value=int(daily["Cases"].iloc[-3]))
    lag7 = c.number_input("Cases 7 days ago", min_value=0, value=int(daily["Cases"].iloc[-7]))

    if st.button("Estimate tomorrow", type="primary"):
        recent = daily["Cases"].tail(14).values
        row = {
            "Lag_Cases_1": lag1,
            "Lag_Cases_3": lag3,
            "Lag_Cases_7": lag7,
            "Lag_Deaths_1": float(daily["Deaths"].iloc[-1]),
            "Lag_Recovered_1": float(daily["Recovered"].iloc[-1]),
            "Roll_Mean_7": float(np.mean(recent[-7:])),
            "Roll_Mean_14": float(np.mean(recent)),
            "Roll_Std_7": float(np.std(recent[-7:], ddof=1)),
            "Roll_Max_14": float(np.max(recent)),
            "Cumul_Cases_Lag": float(daily["Cases"].sum()),
            "DayOfWeek_Num": int(daily["Date"].iloc[-1].dayofweek),
            "Month": int(daily["Date"].iloc[-1].month),
            "CFR_Lag_1": float(daily["Deaths"].sum() / daily["Cases"].sum() * 100),
            "Growth_Lag_1": float(daily["Cases"].pct_change().iloc[-1] * 100),
        }
        X = pd.DataFrame([row])[features]
        learner = float(model.predict(X)[0])

        k1, k2, k3 = st.columns(3)
        k1.metric("Naive (tomorrow = today)", f"{lag1:,.0f}", "best hold-out model")
        k2.metric("7-day average", f"{np.mean(recent[-7:]):,.0f}", "smoothed recent pace")
        k3.metric("Trained learner", f"{learner:,.0f}", "underperforms the naive")

        if learner < 0.5 * lag1 or learner > 2.0 * lag1:
            st.info(
                f"The learner returned {learner:,.0f}, which is more than 2x away from yesterday's "
                f"{lag1:,}. That is expected: `Cumul_Cases_Lag` is {row['Cumul_Cases_Lag']:,.0f}, above "
                f"the largest value it saw in training ({79000:,}), so the model is extrapolating "
                "outside its range. Trust the naive or the 7-day average instead."
            )

# ------------------------------------------------------------------ data tab
with tab_data:
    st.write(f"Filtered rows: **{len(view):,}**")
    show = ["Date", "Province", "City", "Travel_history", "Cases", "Deaths", "Recovered"]
    st.dataframe(view[show].sort_values(["Date", "Province"]).head(2000),
                 use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV",
                       view[show].to_csv(index=False).encode("utf-8"),
                       "covid19_filtered.csv", "text/csv")

st.divider()
st.caption("Data: Pakistan COVID-19 tracking, 26 Feb – 03 Jun 2020 · "
           "Built with pandas, scikit-learn, statsmodels and Plotly.")
