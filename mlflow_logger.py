import os,json,pickle,numpy as np,pandas as pd,mlflow,mlflow.xgboost
from mlflow.models.signature import infer_signature

ARTIFACTS_DIR=r"C:\Users\vaibh\OneDrive\Desktop\Workstation\Internship\artifacts"
MLFLOW_DB=r"C:\Users\vaibh\OneDrive\Desktop\Workstation\Internship\mlflow_fresh.db"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("nifty_direction_classifier")

def load_artifacts():
    sc=pickle.load(open(os.path.join(ARTIFACTS_DIR,"scaler.pkl"),"rb"))
    mdl=pickle.load(open(os.path.join(ARTIFACTS_DIR,"model.pkl"),"rb"))
    oos_m=json.load(open(os.path.join(ARTIFACTS_DIR,"oos_metrics.json")))
    bt_m=json.load(open(os.path.join(ARTIFACTS_DIR,"bt_metrics.json")))
    xgb_p=json.load(open(os.path.join(ARTIFACTS_DIR,"xgb_params.json")))
    features=json.load(open(os.path.join(ARTIFACTS_DIR,"features_list.json")))
    fold_df=pd.read_csv(os.path.join(ARTIFACTS_DIR,"wf_folds.csv"))
    df_oos=pd.read_csv(os.path.join(ARTIFACTS_DIR,"oos_predictions.csv"),parse_dates=["date"])
    return sc,mdl,oos_m,bt_m,xgb_p,features,fold_df,df_oos

def log_folds(fold_df):
    for _,row in fold_df.iterrows():
        with mlflow.start_run(run_name=f"wf_fold_{int(row['fold'])}",nested=True):
            mlflow.log_params({
                "fold":int(row["fold"]),
                "train_end":str(row["train_end"]),
                "val_start":str(row["val_start"]),
                "val_end":str(row["val_end"])
            })
            mlflow.log_metrics({
                "auc":float(row["auc"]),
                "bal_acc":float(row["bal_acc"]),
                "acc":float(row["acc"])
            })

def log_model(sc,mdl,df_oos,features):
    print("Logging XGBoost model...")
    X_sample=sc.transform(df_oos[features].head(5).values)
    predictions=mdl.predict(X_sample)
    signature=infer_signature(X_sample,predictions)
    mlflow.xgboost.log_model(mdl,artifact_path="xgb_model",signature=signature)

def log_final(sc,mdl,oos_m,bt_m,xgb_p,features,fold_df,df_oos):
    with mlflow.start_run(run_name="final_model_oos") as run:
        safe_params={k:str(v) for k,v in xgb_p.items()}
        safe_params.update({
            "features":str(features),
            "n_features":str(len(features)),
            "oos_start":str(df_oos["date"].min().date()),
            "oos_end":str(df_oos["date"].max().date()),
            "buy_threshold":str(bt_m["buy_threshold"]),
            "sell_threshold":str(bt_m["sell_threshold"]),
            "exit_bar":str(bt_m["exit_bar"]),
            "baseline_majority_class":str(oos_m["baseline_majority_class"])
        })

        mlflow.log_params(safe_params)

        mlflow.log_metrics({
            "oos_accuracy":float(oos_m["accuracy"]),
            "oos_precision":float(oos_m["precision"]),
            "oos_recall":float(oos_m["recall"]),
            "oos_f1":float(oos_m["f1"]),
            "oos_auc_mean":float(oos_m["auc_mean"]),
            "oos_auc_ci_lo":float(oos_m["auc_ci_lo"]),
            "oos_auc_ci_hi":float(oos_m["auc_ci_hi"]),
            "oos_bal_acc_mean":float(oos_m["bal_acc_mean"]),
            "oos_bal_acc_ci_lo":float(oos_m["bal_acc_ci_lo"]),
            "oos_bal_acc_ci_hi":float(oos_m["bal_acc_ci_hi"]),
            "baseline_majority_acc":float(oos_m["baseline_majority_acc"]),
            "baseline_persistence_acc":float(oos_m["baseline_persistence_acc"]),
            "bt_sharpe":float(bt_m["sharpe"]),
            "bt_max_drawdown":float(bt_m["max_drawdown"]),
            "bt_hit_rate":float(bt_m["hit_rate"]),
            "bt_return_pct":float(bt_m["return_pct"]),
            "bt_num_trades":float(bt_m["num_trades"]),
            "wf_mean_auc":float(fold_df["auc"].mean()),
            "wf_std_auc":float(fold_df["auc"].std()),
            "wf_mean_bal_acc":float(fold_df["bal_acc"].mean()),
            "wf_std_bal_acc":float(fold_df["bal_acc"].std()),
            "wf_n_folds":float(len(fold_df))
        })

        log_model(sc,mdl,df_oos,features)

        artifact_files=[
            "scaler.pkl",
            "model.pkl",
            "oos_metrics.json",
            "bt_metrics.json",
            "wf_folds.csv",
            "oos_predictions.csv",
            "features_final.csv",
            "xgb_params.json",
            "features_list.json"
        ]

        for fname in artifact_files:
            fpath=os.path.join(ARTIFACTS_DIR,fname)
            if os.path.exists(fpath):
                mlflow.log_artifact(fpath,artifact_path="artifacts")

        log_folds(fold_df)

        print(f"Run logged successfully | Run ID: {run.info.run_id}")

if __name__=="__main__":
    sc,mdl,oos_m,bt_m,xgb_p,features,fold_df,df_oos=load_artifacts()
    log_final(sc,mdl,oos_m,bt_m,xgb_p,features,fold_df,df_oos)
    print(f"MLflow UI -> mlflow ui --backend-store-uri sqlite:///{MLFLOW_DB}")