from pathlib import Path
import json,pandas as pd,numpy as np,mlflow,xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from logger import get_logger
from data_preprocessing import build_dataset
from feature_engineering import create_features,FEATURES

logger=get_logger(__name__)

PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT/"data"/"csv"
ARTIFACTS_DIR=PROJECT_ROOT/"artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True,parents=True)

mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT/'mlflow.db').as_posix()}")
mlflow.set_experiment("nifty_walkforward_validation")

OOS_START_DATE="2025-07-01"
TRAIN_WINDOW=252
TEST_WINDOW=63
STEP_SIZE=21

XGB_PARAMS={"objective":"binary:logistic","eval_metric":"auc","n_estimators":700,"learning_rate":0.015,"max_depth":3,"min_child_weight":8,"subsample":0.7,"colsample_bytree":0.7,"gamma":1.5,"reg_alpha":0.5,"reg_lambda":3.0,"random_state":42,"n_jobs":-1}

def metrics(y,yh,yp):
 return{"accuracy":float(accuracy_score(y,yh)),"balanced_accuracy":float(balanced_accuracy_score(y,yh)),"precision":float(precision_score(y,yh,zero_division=0)),"recall":float(recall_score(y,yh,zero_division=0)),"f1":float(f1_score(y,yh,zero_division=0)),"auc":float(roc_auc_score(y,yp))}

def walk_forward():

 logger.info("Loading dataset")
 df=build_dataset(DATA_DIR)
 df=create_features(df)
 df=df[df["date"]<OOS_START_DATE].reset_index(drop=True)

 miss=[f for f in FEATURES if f not in df.columns]
 if miss:raise ValueError(f"Missing features: {miss}")

 fold=1
 start=TRAIN_WINDOW
 folds, preds=[] ,[]

 while start+TEST_WINDOW<=len(df):

  tr=df.iloc[:start]
  te=df.iloc[start:start+TEST_WINDOW]

  Xtr,Ytr=tr[FEATURES],tr["target"]
  Xte,Yte=te[FEATURES],te["target"]

  scaler=StandardScaler()
  Xtr=scaler.fit_transform(Xtr)
  Xte=scaler.transform(Xte)

  model=xgb.XGBClassifier(**XGB_PARAMS)
  model.fit(Xtr,Ytr,verbose=False)

  prob=model.predict_proba(Xte)[:,1]
  pred=(prob>0.55).astype(int)

  m=metrics(Yte,pred,prob)
  m.update({"fold":fold})

  folds.append(m)

  preds.append(pd.DataFrame({"date":te["date"],"actual":Yte,"pred":pred,"prob":prob,"close":te["close"],"fold":fold}))

  logger.info(f"Fold {fold} | AUC {m['auc']:.4f} | BAL {m['balanced_accuracy']:.4f}")

  fold+=1
  start+=STEP_SIZE

 fold_df=pd.DataFrame(folds)
 pred_df=pd.concat(preds)

 summary={
 "mean_auc":float(fold_df["auc"].mean()),
 "std_auc":float(fold_df["auc"].std()),
 "mean_acc":float(fold_df["accuracy"].mean()),
 "mean_bal_acc":float(fold_df["balanced_accuracy"].mean()),
 "mean_f1":float(fold_df["f1"].mean()),
 "n_folds":len(fold_df)
 }

 fold_df.to_csv(ARTIFACTS_DIR/"wf_folds.csv",index=False)
 pred_df.to_csv(ARTIFACTS_DIR/"wf_predictions.csv",index=False)
 json.dump(summary,open(ARTIFACTS_DIR/"wf_summary.json","w"),indent=2)

 logger.info("Logging MLflow")

 with mlflow.start_run(run_name="walkforward"):

  mlflow.log_params(XGB_PARAMS)
  mlflow.log_param("features",FEATURES)
  mlflow.log_param("train_window",TRAIN_WINDOW)
  mlflow.log_param("step",STEP_SIZE)

  for k,v in summary.items():
   mlflow.log_metric(k,v)

  mlflow.log_artifact(str(ARTIFACTS_DIR/"wf_folds.csv"))
  mlflow.log_artifact(str(ARTIFACTS_DIR/"wf_predictions.csv"))
  mlflow.log_artifact(str(ARTIFACTS_DIR/"wf_summary.json"))

 logger.info(f"AUC {summary['mean_auc']:.4f}")
 logger.info(f"BAL {summary['mean_bal_acc']:.4f}")

 print(f"Run MLflow UI with:\nmlflow ui --backend-store-uri sqlite:///{(PROJECT_ROOT/'mlflow.db').as_posix()}")

 return fold_df,pred_df,summary

if __name__=="__main__":
 walk_forward()