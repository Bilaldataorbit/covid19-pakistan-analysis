markdown
# COVID-19 Pakistan Analysis Portfolio

Complete data analysis and machine learning project on Pakistan's COVID-19 data (Feb-Jun 2020).

## 📊 Project Overview

- **Records:** 2,798
- **Date Range:** Feb 26 - Jun 3, 2020
- **Provinces:** 8
- **Cities:** 125

## 🎯 Key Findings

| Metric | Value |
|--------|-------|
| Total Cases | 83,986 |
| Total Deaths | 1,728 |
| Total Recovered | 24,754 |
| Case Fatality Rate | 2.06% |
| Recovery Rate | 29.47% |

## 📈 Descriptive Statistics (Cases)

| Statistic | Value |
|-----------|-------|
| Mean | 30.02 |
| Variance | 16,605.33 |
| Std Dev | 128.86 |
| Median | 2 |
| Max | 1,639 |

## 🤖 Machine Learning Results

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **Gradient Boosting** 🥇 | **0.7518** | 42.50 | 14.64 |
| Random Forest 🥈 | 0.7353 | 43.89 | 13.73 |
| Linear Regression 🥉 | 0.7251 | 44.72 | 18.65 |

## 📁 Project Structure
Covid-19-portfolio/
├── data/
│ ├── raw/ # Original CSV
│ └── processed/ # Cleaned data
├── notebooks/ # Jupyter analysis
├── outputs/
│ └── figures/ # Charts
├── models/ # Trained ML models
├── requirements.txt
├── .gitignore
└── README.md

text

## 🛠️ Technologies Used

- Python 3.14
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- SciPy
- Jupyter Notebook

## 🚀 How to Run

```bash
# Clone the repo
git clone <your-repo-url>
cd Covid-19-portfolio

# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/COVID_19_analysis.ipynb
👤 Author
Bilal Raza

📅 Date
October 2026

