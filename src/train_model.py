from pathlib import Path
import json,pickle,mlflow,mlflow.xgboost,pandas as pd,xgboost as xgb
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,balanced_accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from logger import get_logger
from data_preprocessing import build_dataset
from feature_engineering import create_features,FEATURES

logger=get_logger(__name__)

PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT/"data"/"csv"
ARTIFACTS_DIR=PROJECT_ROOT/"artifacts"
ARTIFACTS_DIR.mkdir(parents=True,exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT/'mlflow.db').as_posix()}")
mlflow.set_experiment("nifty_direction_classifier")

OOS_START_DATE="2025-07-01"

XGB_PARAMS={"objective":"binary:logistic","eval_metric":"logloss","n_estimators":400,"learning_rate":0.02,"max_depth":4,"min_child_weight":6,"subsample":0.8,"colsample_bytree":0.8,"gamma":1,"reg_alpha":0.1,"reg_lambda":1.0,"random_state":42,"n_jobs":-1}

def compute_metrics(y_true,y_pred,y_prob):
 return{"accuracy":float(accuracy_score(y_true,y_pred)),"balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),"precision":float(precision_score(y_true,y_pred,zero_division=0)),"recall":float(recall_score(y_true,y_pred,zero_division=0)),"f1":float(f1_score(y_true,y_pred,zero_division=0)),"auc":float(roc_auc_score(y_true,y_prob)),"confusion_matrix":confusion_matrix(y_true,y_pred).tolist()}

def compute_baselines(y):
 m=int(y.mean()>=0.5)
 return{"majority_accuracy":float((y==m).mean()),"persistence_accuracy":float((y.iloc[1:].values==y.iloc[:-1].values).mean())}

def train_model():

 logger.info("Loading dataset")
 df=build_dataset(DATA_DIR)

 logger.info("Creating features")
 df=create_features(df)

 df=df.sort_values("date").reset_index(drop=True)

 train_df=df[df["date"]<OOS_START_DATE]
 test_df=df[df["date"]>=OOS_START_DATE]

 X_train,y_train=train_df[FEATURES],train_df["target"]
 X_test,y_test=test_df[FEATURES],test_df["target"]

 base=compute_baselines(y_test)

 logger.info(f"Majority baseline accuracy: {base['majority_accuracy']:.4f}")
 logger.info(f"Persistence baseline accuracy: {base['persistence_accuracy']:.4f}")

 scaler=StandardScaler()
 X_train=scaler.fit_transform(X_train)
 X_test=scaler.transform(X_test)

 logger.info("Training model")
 model=xgb.XGBClassifier(**XGB_PARAMS)
 model.fit(X_train,y_train)

 y_prob=model.predict_proba(X_test)[:,1]
 y_pred=(y_prob>=0.5).astype(int)

 metrics=compute_metrics(y_test,y_pred,y_prob)
 metrics["majority_baseline_accuracy"]=base["majority_accuracy"]
 metrics["persistence_baseline_accuracy"]=base["persistence_accuracy"]

 logger.info(f"Accuracy          : {metrics['accuracy']:.4f}")
 logger.info(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
 logger.info(f"AUC               : {metrics['auc']:.4f}")

 with open(ARTIFACTS_DIR/"model.pkl","wb") as f:pickle.dump(model,f)
 with open(ARTIFACTS_DIR/"scaler.pkl","wb") as f:pickle.dump(scaler,f)
 with open(ARTIFACTS_DIR/"features.json","w") as f:json.dump(FEATURES,f)
 with open(ARTIFACTS_DIR/"metrics.json","w") as f:json.dump(metrics,f)

 logger.info("Logging to MLflow")

 with mlflow.start_run(run_name="xgb_oos_validation"):

  mlflow.log_params(XGB_PARAMS)
  mlflow.log_param("features",FEATURES)
  mlflow.log_param("n_features",len(FEATURES))
  mlflow.log_param("oos_start_date",OOS_START_DATE)

  for k,v in metrics.items():
   if k!="confusion_matrix":
    mlflow.log_metric(k,v)

  mlflow.xgboost.log_model(
   model,
   name="model",
   pip_requirements=["xgboost","scikit-learn","pandas","numpy","mlflow"]
  )

  mlflow.log_artifact(str(ARTIFACTS_DIR/"model.pkl"))
  mlflow.log_artifact(str(ARTIFACTS_DIR/"scaler.pkl"))
  mlflow.log_artifact(str(ARTIFACTS_DIR/"features.json"))
  mlflow.log_artifact(str(ARTIFACTS_DIR/"metrics.json"))

 logger.info("Done")
 return model,scaler,metrics

if __name__=="__main__":
 train_model()