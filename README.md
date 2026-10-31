 ⚡ Power Load Prediction using Machine Learning

Predicting electrical **power consumption** from electrical and environmental measurements using **Linear Regression** and **Ridge Regression** (scikit-learn).

> Academic project — *303-05 Fundamentals of Machine Learning (PR)*, SYBCA-AI Semester III, Sutex Bank College of Computer Application & Science, Surat, Gujarat (2026–2027). Faculty: Dr. Hiral A. Patel.

## 📌 Overview

Power utilities need to estimate load to manage energy and balance the grid. This project takes a tabular dataset of voltage, current, power factor, reactive power, voltage fluctuation and humidity readings, cleans it, explores it, and trains regression models to predict `Power_Consumption`.

## 📊 Results

Evaluated on a held-out 20 % test set (`random_state=42`), 348 rows total (278 train / 70 test):

| Model | MAE | RMSE | R² (test) | R² (5-fold CV) |
|---|---|---|---|---|
| Linear Regression | 0.0449 | 0.0690 | **0.99940** | 0.99952 |
| Ridge Regression (α=1) | 0.0451 | 0.0695 | 0.99939 | 0.99952 |

![Actual vs predicted](images/Actual_vs_Predicted.png)

**Key finding:** `Current (A)` is almost perfectly correlated with power consumption (r ≈ 0.999), followed by `Reactive Power (kVAR)` (r ≈ 0.76). This is physically expected (P ≈ V·I·cosφ), which is why a simple linear model works so well.

| Distribution | Correlation matrix |
|---|---|
| ![hist](images/Histogram.png) | ![heatmap](images/Heatmap.png) |

## 🗂️ Dataset

`data/dataset.csv` — 348 rows × 12 columns, no missing values, no duplicates (all from year 2024).

| Column | Description |
|---|---|
| Year | Year of observation |
| Voltage (V) | Voltage in volts |
| Current (A) | Current in amperes |
| **Power_Consumption** | **Target variable** |
| Reactive Power (kVAR) | Non-active power in the AC circuit |
| Power Factor | Ratio of real to apparent power |
| Solar Power (kW) / Wind Power (kW) | Renewable generation |
| Voltage Fluctuation (%) | Variation in voltage level |
| Overload Condition | 0 / 1 flag |
| Temperature (°C), Humidity (%) | Weather conditions |

Features used by the model: Voltage, Current, Power Factor, Voltage Fluctuation, Humidity, Reactive Power.

## 🧪 Methodology

1. Data inspection (shape, dtypes, missing, duplicates, statistics)
2. Outlier handling on `Power_Consumption` with the IQR rule (no rows were flagged)
3. EDA: histogram, box plots, correlation heatmap, [Sweetviz report](docs/SWEETVIZ_REPORT.html)
4. 80/20 train–test split (`random_state=42`)
5. Train Linear & Ridge Regression, evaluate with MAE, MSE, RMSE, R², 5-fold cross-validation

## 📁 Project structure

```
Power_Load_Prediction/
├── data/dataset.csv
├── src/
│   ├── config.py        # paths, feature list, constants
│   ├── eda.py           # EDA plots + outlier handling
│   ├── train.py         # train / evaluate / save model
│   └── predict.py       # predict for a new reading
├── images/              # generated plots
├── docs/
│   ├── Project_Report.pdf
│   └── SWEETVIZ_REPORT.html
├── models/              # saved model + metrics (generated)
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚀 Getting started

```bash
git clone https://github.com/<your-username>/Power_Load_Prediction.git
cd Power_Load_Prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/eda.py      # create plots
python src/train.py    # train + evaluate + save model
python src/predict.py  # sample prediction
```

## ⚠️ Limitations

- Small dataset (348 rows) from a single year; results may not generalise to real grid data.
- Rows look independent (no timestamp), so this is a regression on instantaneous readings, **not** time-series forecasting.
- Because current drives power almost linearly, the high R² reflects physics more than model sophistication.

## 🔮 Future work

Add time-of-day / day-of-week / holiday features, try Random Forest, Gradient Boosting or neural networks, and wrap the model in a small web app (Streamlit/Flask).

## 👥 Authors

- Yug Jagdishbhai Bhimani (Roll 576)
- Aryan Vallabhbhai Paladiya (Roll 552)

## 📄 License

Released under the [MIT License](LICENSE).
