"""
COVID-19 Pakistan Analysis — Interactive Dashboard
Streamlit web application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="COVID-19 Pakistan Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PATHS (Auto-detect for both local & cloud)
# ==========================================
# app.py is at: <project_root>/app/app.py
# Project root is one level up from app/
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_PATH, 'data', 'processed')
MODELS_PATH = os.path.join(BASE_PATH, 'models')

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_PATH, 'PK_COVID_19_cleaned.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def load_model():
    with open(os.path.join(MODELS_PATH, 'covid_cases_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_PATH, 'province_encoder.pkl'), 'rb') as f:
        encoder = pickle.load(f)
    with open(os.path.join(MODELS_PATH, 'model_features.pkl'), 'rb') as f:
        features = pickle.load(f)
    return model, encoder, features

df = load_data()
model, encoder, features = load_model()

# ==========================================
# HEADER
# ==========================================
st.title("🦠 COVID-19 Pakistan Dashboard")
st.markdown("**Interactive Analysis | Feb 26 - Jun 3, 2020**")
st.markdown("---")

# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.header("🔍 Filters")

min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

provinces = ['All'] + sorted(df['Province'].unique().tolist())
selected_province = st.sidebar.selectbox("Select Province", provinces)

travel_histories = ['All'] + sorted(df['Travel_history'].unique().tolist())
selected_travel = st.sidebar.selectbox("Select Travel History", travel_histories)

# ==========================================
# FILTER DATA
# ==========================================
filtered_df = df.copy()

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df['Date'].dt.date >= start_date) &
        (filtered_df['Date'].dt.date <= end_date)
    ]

if selected_province != 'All':
    filtered_df = filtered_df[filtered_df['Province'] == selected_province]

if selected_travel != 'All':
    filtered_df = filtered_df[filtered_df['Travel_history'] == selected_travel]

# ==========================================
# KPI CARDS
# ==========================================
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

total_cases = filtered_df['Cases'].sum()
total_deaths = filtered_df['Deaths'].sum()
total_recovered = filtered_df['Recovered'].sum()
active_cases = total_cases - total_deaths - total_recovered

col1.metric("Total Cases", f"{total_cases:,}",
            delta=f"{filtered_df['Cases'].mean():.1f} avg/day")
col2.metric("Total Deaths", f"{total_deaths:,}",
            delta=f"{(total_deaths/total_cases*100 if total_cases>0 else 0):.2f}% CFR")
col3.metric("Total Recovered", f"{total_recovered:,}",
            delta=f"{(total_recovered/total_cases*100 if total_cases>0 else 0):.2f}% rate")
col4.metric("Active Cases", f"{active_cases:,}")

st.markdown("---")

# ==========================================
# TIME SERIES CHART
# ==========================================
st.subheader("📈 Daily Trend")

daily = filtered_df.groupby('Date')[['Cases', 'Deaths', 'Recovered']].sum().reset_index()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily['Date'], daily['Cases'], color='steelblue', label='Cases', linewidth=1.5)
ax.plot(daily['Date'], daily['Deaths'], color='crimson', label='Deaths', linewidth=1.5)
ax.plot(daily['Date'], daily['Recovered'], color='green', label='Recovered', linewidth=1.5)
ax.set_xlabel('Date')
ax.set_ylabel('Count')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ==========================================
# PROVINCE & TRAVEL
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🗺️ Cases by Province")
    province_cases = filtered_df.groupby('Province')['Cases'].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(province_cases.index, province_cases.values, color='steelblue', edgecolor='black')
    ax.set_xlabel('Total Cases')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("🛫 Cases by Travel History")
    travel_cases = filtered_df.groupby('Travel_history')['Cases'].sum().sort_values(ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(travel_cases.values, labels=travel_cases.index, autopct='%1.1f%%', startangle=90)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# ==========================================
# DISTRIBUTION
# ==========================================
st.subheader("📊 Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(filtered_df['Cases'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(filtered_df['Cases'].mean(), color='red', linestyle='--',
               label=f'Mean = {filtered_df["Cases"].mean():.1f}')
    ax.axvline(filtered_df['Cases'].median(), color='green', linestyle='--',
               label=f'Median = {filtered_df["Cases"].median():.1f}')
    ax.set_xlabel('Cases')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Province', y='Cases', data=filtered_df, ax=ax)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# ==========================================
# ML PREDICTION
# ==========================================
st.subheader("🤖 Predict Cases (Machine Learning)")
st.markdown("Neeche input dein aur model se prediction lein:")

col1, col2, col3 = st.columns(3)

with col1:
    input_deaths = st.number_input("Deaths", min_value=0, max_value=100, value=5)
    input_recovered = st.number_input("Recovered", min_value=0, max_value=2000, value=50)

with col2:
    input_local = st.selectbox("Is Local Transmission?", [0, 1],
                                format_func=lambda x: "Yes" if x == 1 else "No")
    input_tableeghi = st.selectbox("Is Tableeghi Jamaat?", [0, 1],
                                    format_func=lambda x: "Yes" if x == 1 else "No")

with col3:
    input_month = st.slider("Month", 1, 12, 6)
    input_dayofweek = st.slider("Day of Week (0=Mon)", 0, 6, 2)

selected_pred_province = st.selectbox("Province", sorted(df['Province'].unique()))

if st.button("🔮 Predict Cases", type="primary"):
    province_encoded = encoder.transform([selected_pred_province])[0]

    input_data = pd.DataFrame({
        'Deaths': [input_deaths],
        'Recovered': [input_recovered],
        'Is_Local': [input_local],
        'Is_Tableeghi': [input_tableeghi],
        'Month': [input_month],
        'DayOfWeek': [input_dayofweek],
        'Province_Encoded': [province_encoded]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"### 📊 Predicted Cases: **{prediction:.0f}**")
    st.info(f"📍 Province: {selected_pred_province}")

st.markdown("---")

# ==========================================
# ARIMA FORECAST SECTION
# ==========================================
st.subheader("📈 30-Day ARIMA Forecast")

@st.cache_data
def load_forecast():
    daily_ts = df.groupby('Date')[['Cases']].sum().reset_index()
    daily_ts = daily_ts.set_index('Date')
    daily_ts = daily_ts.asfreq('D').ffill()

    model_arima = ARIMA(daily_ts['Cases'], order=(7, 1, 1)).fit()
    forecast = model_arima.get_forecast(steps=30)

    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int(alpha=0.05)

    last_date = daily_ts.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                  periods=30, freq='D')

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': forecast_mean.values.clip(min=0),
        'Lower_CI': forecast_ci.iloc[:, 0].values.clip(min=0),
        'Upper_CI': forecast_ci.iloc[:, 1].values.clip(min=0)
    })

    return daily_ts, forecast_df

daily_ts, forecast_data = load_forecast()

# Forecast chart
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily_ts.index, daily_ts['Cases'], color='steelblue',
        label='Historical', linewidth=1.5)
ax.plot(forecast_data['Date'], forecast_data['Forecast'],
        color='red', label='30-Day Forecast', linewidth=2, linestyle='--')
ax.fill_between(forecast_data['Date'],
                forecast_data['Lower_CI'],
                forecast_data['Upper_CI'],
                color='red', alpha=0.2, label='95% Confidence Interval')
ax.axvline(x=daily_ts.index.max(), color='gray', linestyle=':', linewidth=2,
           label='Forecast Start')

ax.set_xlabel('Date')
ax.set_ylabel('Daily Cases')
ax.set_title('COVID-19 Pakistan: 30-Day Forecast (ARIMA)', fontsize=14)
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# Forecast summary
col1, col2, col3 = st.columns(3)
col1.metric("Total Predicted (30d)", f"{forecast_data['Forecast'].sum():,.0f}")
col2.metric("Avg Daily Cases", f"{forecast_data['Forecast'].mean():,.0f}")
col3.metric("Peak Predicted Day", f"{forecast_data['Forecast'].max():,.0f}")

st.markdown("---")

# ==========================================
# FOOTER
# ==========================================
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Made with ❤️ using Streamlit | Data: Feb 26 - Jun 3, 2020</p>
    </div>
    """,
    unsafe_allow_html=True
)