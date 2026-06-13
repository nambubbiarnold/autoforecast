# AutoForecast Pro

Production-grade multi-SKU forecasting app. Upload your Excel file and run.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Installation (detailed)

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
streamlit run app.py
```

## Expected Excel Format

| ds         | y   | sku_id |
|------------|-----|--------|
| 2024-01-01 | 100 | A      |
| 2024-02-01 | 110 | A      |
| 2024-01-01 | 200 | B      |

- **ds** — date column (any parseable format)
- **y** — numeric value (sales, demand, etc.)
- **sku_id** — optional grouping column (product, store, region)

Column names are auto-detected but can be remapped in the sidebar.
If no SKU column is present, the entire file is treated as a single series.

## Features

| Feature | Detail |
|---|---|
| File formats | `.xlsx`, `.xls`, `.csv` |
| Models | ARIMA, Exp. Smoothing, XGBoost, Prophet, Theta |
| Cleaning | None, Hampel filter, DWT, SSA |
| Optimization | Random search, Optuna (Bayesian), Grid search |
| Auto-selection | Best cleaning+model per SKU by validation error |
| Metrics | SMAPE, MAE, RMSE, MAPE |
| Downloads | Forecasts CSV, pipeline results CSV, reproducibility JSON |

## Optional: Prophet & advanced models

```bash
pip install prophet               # Facebook Prophet
pip install neuralprophet         # NeuralProphet
pip install tensorflow            # LSTM
```

These are optional — the app runs without them using ARIMA, Exp. Smoothing, XGBoost, and Theta.

## Docker

```bash
docker build -t autoforecast .
docker run -p 8501:8501 autoforecast
# Open http://localhost:8501
```
