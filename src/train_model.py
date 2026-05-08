from pathlib import Path
import json
import pickle

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler

from logger import get_logger
from data_preprocessing import build_dataset, FEATURES

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
mlflow.set_experiment("nifty_direction_classifier")

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 300,
    "random_state": 42
}

OOS_START_DATE = "2025-07-01"

def compute_metrics(y_true, y_pred, y_prob):

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "auc": float(
            roc_auc_score(y_true, y_prob)
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred
        ).tolist()
    }

    return metrics

def compute_baselines(y_true):

    majority_class = int(y_true.mean() >= 0.5)

    majority_accuracy = float(
        (y_true == majority_class).mean()
    )

    persistence_accuracy = float(
        (y_true.iloc[1:].values == y_true.iloc[:-1].values).mean()
    )

    return {
        "majority_class": majority_class,
        "majority_accuracy": majority_accuracy,
        "persistence_accuracy": persistence_accuracy
    }

def train_model():

    logger.info("Loading processed dataset")

    logger.info(f"Using data directory: {DATA_DIR}")

    df = build_dataset(DATA_DIR)

    logger.info("Splitting IS vs OOS")

    train_df = df[df["date"] < OOS_START_DATE].copy()

    test_df = df[df["date"] >= OOS_START_DATE].copy()

    if len(test_df) == 0:

        raise ValueError(
            f"No OOS rows found after {OOS_START_DATE}"
        )

    logger.info(f"Train rows: {len(train_df)}")
    logger.info(f"Test rows : {len(test_df)}")

    X_train = train_df[FEATURES]
    y_train = train_df["target"]

    X_test = test_df[FEATURES]
    y_test = test_df["target"]

    baselines = compute_baselines(y_test)

    logger.info(
        f"Majority baseline accuracy: "
        f"{baselines['majority_accuracy']:.4f}"
    )

    logger.info(
        f"Persistence baseline accuracy: "
        f"{baselines['persistence_accuracy']:.4f}"
    )

    logger.info("Scaling features")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    logger.info("Training XGBoost model")

    model = xgb.XGBClassifier(**XGB_PARAMS)

    model.fit(X_train_scaled, y_train)
    logger.info("Generating predictions")

    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    y_pred = (y_prob > 0.5).astype(int)

    metrics = compute_metrics(
        y_test,
        y_pred,
        y_prob
    )

    metrics["majority_baseline_accuracy"] = float(
        baselines["majority_accuracy"]
    )

    metrics["persistence_baseline_accuracy"] = float(
        baselines["persistence_accuracy"]
    )

    logger.info(f"Accuracy           : {metrics['accuracy']:.4f}")

    logger.info(
        f"Balanced Accuracy  : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    logger.info(f"Precision          : {metrics['precision']:.4f}")

    logger.info(f"Recall             : {metrics['recall']:.4f}")

    logger.info(f"F1 Score           : {metrics['f1']:.4f}")

    logger.info(f"AUC Score          : {metrics['auc']:.4f}")

    oos_df = pd.DataFrame({

        "date": test_df["date"].values,

        "actual": y_test.values,

        "prediction": y_pred,

        "probability": y_prob,

        "open": test_df["open"].values,

        "high": test_df["high"].values,

        "low": test_df["low"].values,

        "close": test_df["close"].values,

        "volume": test_df["volume"].values,

        "ema50": test_df["ema50"].values
    })

    logger.info("Saving artifacts")

    with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:

        pickle.dump(model, f)

    with open(ARTIFACTS_DIR / "scaler.pkl", "wb") as f:

        pickle.dump(scaler, f)

    with open(ARTIFACTS_DIR / "features.json", "w") as f:

        json.dump(FEATURES, f, indent=4)

    with open(ARTIFACTS_DIR / "xgb_params.json", "w") as f:

        json.dump(XGB_PARAMS, f, indent=4)

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:

        json.dump(metrics, f, indent=4)

    oos_df.to_csv(
        ARTIFACTS_DIR / "oos_predictions.csv",
        index=False
    )

    logger.info("Logging experiment to MLflow")

    with mlflow.start_run(
        run_name="xgboost_oos_validation"
    ):

        mlflow.log_params(XGB_PARAMS)

        mlflow.log_param(
            "n_features",
            len(FEATURES)
        )

        mlflow.log_param(
            "features",
            str(FEATURES)
        )

        mlflow.log_param(
            "oos_start_date",
            OOS_START_DATE
        )

        for key, value in metrics.items():

            if key != "confusion_matrix":

                mlflow.log_metric(key, value)

        mlflow.xgboost.log_model(
            model,
            artifact_path="xgb_model"
        )

        artifact_files = [
            "model.pkl",
            "scaler.pkl",
            "features.json",
            "xgb_params.json",
            "metrics.json",
            "oos_predictions.csv"
        ]

        for file_name in artifact_files:

            mlflow.log_artifact(
                str(ARTIFACTS_DIR / file_name)
            )

    logger.info("Training completed successfully")

    return model, scaler, metrics, oos_df

if __name__ == "__main__":

    train_model()