"""train_model.py file"""
from pathlib import Path
import json, pickle, mlflow, mlflow.xgboost, pandas as pd, xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, balanced_accuracy_score,
                             confusion_matrix)
from logger import get_logger
from data_preprocessing import build_dataset
from feature_engineering import create_features, FEATURES

import warnings
warnings.filterwarnings("ignore")

logger = get_logger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "csv"
ART      = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{(ROOT/'mlflow.db').as_posix()}")
mlflow.set_experiment("nifty_direction_classifier")

OOS = "2025-07-01"

PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "n_estimators":     400,
    "learning_rate":    0.015,
    "max_depth":        2,
    "min_child_weight": 2,
    "subsample":        0.9,
    "colsample_bytree": 0.67,
    "gamma":            4.5,
    "reg_alpha":        1.15,
    "reg_lambda":       2.8,
    "random_state":     42,
    "n_jobs":           -1,
}
# PARAMS = {}

THRESHOLD = 0.55         # unified threshold — same value used in walkforward.py

def metrics(y, p, pr):
    return {
        "accuracy":          float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "precision":         float(precision_score(y, p, zero_division=0)),
        "recall":            float(recall_score(y, p, zero_division=0)),
        "f1":                float(f1_score(y, p, zero_division=0)),
        "auc":               float(roc_auc_score(y, pr)),
        "confusion_matrix":  confusion_matrix(y, p).tolist(),
    }

def baselines(y):
    m = int(y.mean() >= 0.5)
    return {
        "majority":    float((y == m).mean()),
        "persistence": float((y.iloc[1:].values == y.iloc[:-1].values).mean()),
    }

def train():

    # ── 1. build raw merged dataset ──────────────────────────────────────────
    raw = build_dataset(DATA_DIR)

    # ── 2. create features on FULL dataset so rolling windows have complete
    #       history at every row — then split by date ─────────────────────────
    df = create_features(raw)          # target = NaN on last row → dropped inside

    train = df[df["date"] < OOS].copy().sort_values("date").reset_index(drop=True)
    test  = df[df["date"] >= OOS].copy().sort_values("date").reset_index(drop=True)

    X_train, y_train = train[FEATURES], train["target"]
    X_test,  y_test  = test[FEATURES],  test["target"]

    # ── 3. scale — fit ONLY on train, transform both ─────────────────────────
    scaler  = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test  = scaler.transform(X_test)

    # ── 4. baselines (computed on raw OOS labels, no model needed) ───────────
    base = baselines(y_test)
    logger.info(f"Baselines: {base}")

    # ── 5. train ──────────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(Xs_train, y_train)

    # ── 6. evaluate ───────────────────────────────────────────────────────────
    prob = model.predict_proba(Xs_test)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)

    m = metrics(y_test, pred, prob)
    m.update({
        "majority_baseline":    base["majority"],
        "persistence_baseline": base["persistence"],
        "threshold":            THRESHOLD,
    })
    logger.info(m)

    # ── 7. save artifacts ─────────────────────────────────────────────────────
    with open(ART / "model.pkl",   "wb") as f: pickle.dump(model,  f)
    with open(ART / "scaler.pkl",  "wb") as f: pickle.dump(scaler, f)   # backtest needs this
    with open(ART / "features.json", "w") as f: json.dump(FEATURES, f)
    with open(ART / "metrics.json",  "w") as f: json.dump(m, f)

    # save OOS slice with raw (unscaled) feature columns so backtest can
    # reconstruct OHLCV correctly; scaler is applied inside backtest.py
    test[["date", "open", "high", "low", "close", "volume"]
         + FEATURES + ["target"]].to_csv(ART / "oos_predictions.csv", index=False)

    # ── 8. MLflow ─────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="xgb_oos"):
        mlflow.log_params(PARAMS)
        mlflow.log_param("threshold", THRESHOLD)
        for k, v in m.items():
            if k != "confusion_matrix":
                mlflow.log_metric(k, v)
        mlflow.xgboost.log_model(model, "model")

    logger.info("done")

if __name__ == "__main__":
    train()