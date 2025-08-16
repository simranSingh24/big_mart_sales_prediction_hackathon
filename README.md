# Big Mart Sales Forecast — Big Mart Sales III Hackathon

**Competition:** [Big Mart Sales Prediction III](https://www.analyticsvidhya.com/datahack/contest/practice-problem-big-mart-sales-iii/) on Analytics Vidhya, a platform offering a professional setup for showcasing data science skills  [oai_citation:0‡Analytics Vidhya](https://www.analyticsvidhya.com/datahack/contest/practice-problem-big-mart-sales-iii?utm_source=chatgpt.com)

---

##  Overview

This project delivers an end-to-end solution for forecasting **item–outlet level sales** in the Big Mart dataset. It includes exploratory data analysis (EDA), data cleaning, feature engineering, and a robust modeling pipeline using AutoGluon. The design emphasizes **reproducibility**, **generalization**, and adherence to **business constraints** (no negative forecasts; item–outlet granularity preserved).

---

##  Problem Statement

- Forecast `Item_Outlet_Sales` for the test set based on provided historical data.
- Comply with constraints:
  - **No negative predictions**
  - **Maintain Item–Outlet granularity**

---

##  Repository Structure
├── eda_all.ipynb           # Exploratory Data Analysis & data cleaning
├── model_train.ipynb       # Modeling, OOF diagnostics, prediction
├── data                    # Main data folder
├──|── train.csv            # Raw training data
├──|── test.csv             # Raw training data
├──|── cleaned_train.csv     # Cleaned training data
├──|── cleaned_test.csv     # Cleaned training data
├── submission.csv          # Final predictions for submission
└── README.md               # Project documentation

---

##  Approach Summary

### 1. EDA & Data Cleaning
- Inspected dataset schema and target distribution (heavy right skew).
- Evaluated `raw`, `log1p`, and `sqrt` transformations—kept `raw` as baseline.
- Addressed missing/invalid values:
  - `Item_Weight`: imputed using **mode per item**
  - `Item_Visibility = 0`: replaced with **mean per item**
  - `Outlet_Size` (missing): inferred via outlet attributes (Small for specific outlets, others Medium).

### 2. Feature Engineering
- Created `Outlet_Age` (2025 − establishment year) to capture maturity.
- Reserved candidate features (MRP quartile bins, visibility-to-weight ratio, new outlet flag) for iterations.
- Applied consistent preprocessing across train and test by combining and then splitting.

### 3. Modeling Strategy
- Tools: **AutoGluon TabularPredictor**
- Ensemble of multiple algorithms: RF, LightGBM, XGBoost, CatBoost, KNN, Linear Regression, ExtraTrees.
- Training setup: bagging with saved folds for OOF diagnostics, RMSE objective, predictions clipped at zero.

### 4. Experiments & Insights
- **Key finding:** CatBoost often **under-forecasted sales peaks**, likely due to skewed target distribution, regularization, and smoothing.
- Plans for improvement:
  - Apply target transformations (e.g., `log1p`, `sqrt`) for CatBoost specifically.
  - Run feature ablation studies and hyperparameter tuning.
  - Perform residual diagnostics and quantile corrections to address tail bias.

---

##  Results

- Public leaderboard score of 1145.3306496500, rank #146.
- Final `submission.csv` adheres to business rules (non-negative, preserves granularity).

---

##  How to Run

```bash
git clone https://github.com/<your-username>/bigmart-sales-forecast.git
cd bigmart-sales-forecast

# Install dependencies
pip install -r requirements.txt

# Perform EDA
jupyter notebook eda_all.ipynb

# Train model and generate submission
jupyter notebook model_train.ipynb
