from pathlib import Path
import json
import pickle

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
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
mlflow.set_experiment("nifty_walkforward_validation")

OOS_START_DATE = "2025-07-01"

TRAIN_WINDOW = 252
TEST_WINDOW = 63
STEP_SIZE = 63

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

def compute_metrics(y_true, y_pred, y_prob):

    return {
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
        )
    }

def walk_forward():

    logger.info("Loading dataset")

    df = build_dataset(DATA_DIR)

    df = df[df["date"] < OOS_START_DATE].copy()

    logger.info(f"IS rows: {len(df)}")

    X = df[FEATURES]
    y = df["target"]

    fold_results = []
    prediction_rows = []

    start = TRAIN_WINDOW
    fold = 1

    while start + TEST_WINDOW <= len(df):

        train_df = df.iloc[:start]

        test_df = df.iloc[start:start + TEST_WINDOW]

        X_train = train_df[FEATURES]
        y_train = train_df["target"]

        X_test = test_df[FEATURES]
        y_test = test_df["target"]

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)

        X_test_scaled = scaler.transform(X_test)

        model = xgb.XGBClassifier(**XGB_PARAMS)

        model.fit(X_train_scaled, y_train)

        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        y_pred = (y_prob > 0.5).astype(int)

        metrics = compute_metrics(
            y_test,
            y_pred,
            y_prob
        )

        metrics["fold"] = fold

        metrics["train_end"] = str(
            train_df["date"].iloc[-1].date()
        )

        metrics["val_start"] = str(
            test_df["date"].iloc[0].date()
        )

        metrics["val_end"] = str(
            test_df["date"].iloc[-1].date()
        )

        fold_results.append(metrics)

        pred_df = pd.DataFrame({

            "date": test_df["date"].values,

            "actual": y_test.values,

            "prediction": y_pred,

            "probability": y_prob,

            "fold": fold
        })

        prediction_rows.append(pred_df)

        logger.info(
            f"Fold {fold} | "
            f"AUC={metrics['auc']:.4f} | "
            f"BAL_ACC={metrics['balanced_accuracy']:.4f}"
        )

        fold += 1

        start += STEP_SIZE

    fold_df = pd.DataFrame(fold_results)

    pred_df = pd.concat(prediction_rows).reset_index(drop=True)

    wf_summary = {

        "mean_accuracy": float(fold_df["accuracy"].mean()),
        "std_accuracy": float(fold_df["accuracy"].std()),

        "mean_balanced_accuracy": float(
            fold_df["balanced_accuracy"].mean()
        ),

        "std_balanced_accuracy": float(
            fold_df["balanced_accuracy"].std()
        ),

        "mean_precision": float(
            fold_df["precision"].mean()
        ),

        "mean_recall": float(
            fold_df["recall"].mean()
        ),

        "mean_f1": float(
            fold_df["f1"].mean()
        ),

        "mean_auc": float(
            fold_df["auc"].mean()
        ),

        "std_auc": float(
            fold_df["auc"].std()
        ),

        "n_folds": int(len(fold_df))
    }

    fold_df.to_csv(
        ARTIFACTS_DIR / "wf_folds.csv",
        index=False
    )

    pred_df.to_csv(
        ARTIFACTS_DIR / "wf_predictions.csv",
        index=False
    )

    with open(
        ARTIFACTS_DIR / "wf_summary.json",
        "w"
    ) as f:

        json.dump(wf_summary, f, indent=4)

    logger.info("Logging walk-forward run to MLflow")

    with mlflow.start_run(
        run_name="walk_forward_validation"
    ):

        mlflow.log_params(XGB_PARAMS)

        mlflow.log_param(
            "features",
            str(FEATURES)
        )

        mlflow.log_param(
            "n_features",
            len(FEATURES)
        )

        mlflow.log_param(
            "train_window",
            TRAIN_WINDOW
        )

        mlflow.log_param(
            "test_window",
            TEST_WINDOW
        )

        mlflow.log_param(
            "step_size",
            STEP_SIZE
        )

        for key, value in wf_summary.items():

            mlflow.log_metric(key, value)

        mlflow.log_artifact(
            str(ARTIFACTS_DIR / "wf_folds.csv")
        )

        mlflow.log_artifact(
            str(ARTIFACTS_DIR / "wf_predictions.csv")
        )

        mlflow.log_artifact(
            str(ARTIFACTS_DIR / "wf_summary.json")
        )

    logger.info("Walk-forward completed")

    logger.info(
        f"Mean AUC: "
        f"{wf_summary['mean_auc']:.4f}"
    )

    logger.info(
        f"Mean Balanced Accuracy: "
        f"{wf_summary['mean_balanced_accuracy']:.4f}"
    )

    return fold_df, pred_df, wf_summary

if __name__ == "__main__":

    walk_forward()