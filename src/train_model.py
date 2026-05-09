from pathlib import Path
import json, pickle, mlflow, mlflow.xgboost, pandas as pd, xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
from logger import get_logger
from data_preprocessing import build_dataset
from feature_engineering import create_features, FEATURES

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{(ROOT/'mlflow.db').as_posix()}")
mlflow.set_experiment("nifty_direction_classifier")

OOS = "2025-07-01"

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 400,
    "learning_rate": 0.015,
    "max_depth": 2,
    "min_child_weight": 2,
    "subsample": 0.9,
    "colsample_bytree": 0.67,
    "gamma": 4.5,
    "reg_alpha": 1.15,
    "reg_lambda": 2.8,
    "random_state": 42,
    "n_jobs": -1
}

def metrics(y, p, pr):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "auc": float(roc_auc_score(y, pr)),
        "confusion_matrix": confusion_matrix(y, p).tolist()
    }

def baselines(y):
    m = int(y.mean() >= 0.5)
    return {
        "majority": float((y == m).mean()),
        "persistence": float((y.iloc[1:].values == y.iloc[:-1].values).mean())
    }

def train():

    df = build_dataset(DATA_DIR)

    train_raw = df[df["date"] < OOS].copy()
    test_raw = df[df["date"] >= OOS].copy()

    train = create_features(train_raw)
    test = create_features(test_raw)

    train = train.sort_values("date").reset_index(drop=True)
    test = test.sort_values("date").reset_index(drop=True)

    X_train, y_train = train[FEATURES], train["target"]
    X_test, y_test = test[FEATURES], test["target"]

    base = baselines(y_test)

    logger.info(base)

    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    m = metrics(y_test, pred, prob)
    print(m)
    m.update({
        "majority_baseline": base["majority"],
        "persistence_baseline": base["persistence"]
    })

    with open(ART / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(ART / "features.json", "w") as f:
        json.dump(FEATURES, f)

    with open(ART / "metrics.json", "w") as f:
        json.dump(m, f)

    test[["date","close","open","high","low"] + FEATURES + ["target"]].to_csv(ART / "oos_predictions.csv", index=False)

    with mlflow.start_run(run_name="xgb_oos"):

        mlflow.log_params(PARAMS)
        for k, v in m.items():
            if k != "confusion_matrix":
                mlflow.log_metric(k, v)

        mlflow.xgboost.log_model(model, "model")

    logger.info("done")

if __name__ == "__main__":
    train()