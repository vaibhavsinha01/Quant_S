# NIFTY 50 Direction Prediction Classifier
**Quant Singularity — Financial ML Intern Screening | Summer 2026**

---

## Setup

1. Clone the repository
2. Install dependencies

pip install -r requirements.txt

3. Ensure the data files are in `data/csv/`:
   - `nifty50.csv`
   - `banknifty.csv`
   - `indiavix.csv`
   - `starter_features.csv`

---

## Run Order

Run the scripts in this order:


# 1. Train the model and evaluate on OOS window
python src/train_model.py

# 2. Walk-forward validation
python src/walkforward.py

# 3. Backtest on OOS window
python src/backtest.py

---

## MLflow

To view experiment tracking:
mlflow ui --backend-store-uri sqlite:///mlflow.db
Then open `http://localhost:5000` in your browser.

---

## Project Structure
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── walkforward.py
│   ├── backtest.py
│   └── logger.py
├── data/csv/
├── artifacts/
├── mlruns/
├── mlflow.db
├── requirements.txt
└── report.pd

---

## Notes

- Do not run scripts out of order — `train_model.py` must run before `backtest.py`
- All artifacts are saved to the `artifacts/` directory automatically
- The OOS window (July–December 2025) is never touched during training or hyperparameter selection
- The report is stored in report.pdf