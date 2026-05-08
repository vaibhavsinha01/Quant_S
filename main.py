import os,json,pickle,warnings,numpy as np,pandas as pd
from ta.momentum import RSIIndicator
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from backtesting import Backtest,Strategy

warnings.filterwarnings("ignore")

BASE=r"C:\Users\vaibh\OneDrive\Desktop\Workstation\Internship\intern_data"
ARTIFACTS_DIR=r"C:\Users\vaibh\OneDrive\Desktop\Workstation\Internship\artifacts"

os.makedirs(ARTIFACTS_DIR,exist_ok=True)

NIFTY_PATH=os.path.join(BASE,"nifty50.csv")
VIX_PATH=os.path.join(BASE,"indiavix.csv")

OOS_START="2025-07-01"

BUY_THRESHOLD=0.55
SELL_THRESHOLD=0.40
EXIT_BAR=2

FEATURES=[
    "volume_normalized",
    "ma5_smooth_signal",
    "ret_5d",
    "vol_50d",
    "vix_level",
    "vol_20d",
    "ret_1d",
    "close_vs_ma5",
    "high_low_range",
    "dow",
    "ret_overnight",
    "momentum_5_20",
    "ret_zscore"
]

XGB_PARAMS={
    "n_estimators":300,
    "max_depth":4,
    "learning_rate":0.03,
    "subsample":0.8,
    "colsample_bytree":0.8,
    "gamma":1,
    "min_child_weight":5,
    "reg_alpha":0.1,
    "reg_lambda":1.0,
    "random_state":42,
    "n_jobs":-1,
    "use_label_encoder":False,
    "eval_metric":"logloss"
}

def _load(path):
    df=pd.read_csv(path,parse_dates=["date"])
    df.columns=df.columns.str.strip().str.lower()
    df=df.sort_values("date").reset_index(drop=True)
    required_cols=["date","open","high","low","close","volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")
    return df

def build_features():
    nifty=_load(NIFTY_PATH)
    vix=_load(VIX_PATH)
    df=nifty.copy()

    df["ret_1d"]=df["close"].pct_change(1)
    df["ret_5d"]=df["close"].pct_change(5)
    df["ret_overnight"]=(df["open"]-df["close"].shift(1))/df["close"].shift(1)
    df["high_low_range"]=(df["high"]-df["low"])/df["low"]

    for w in [5,20,50]:
        df[f"_ma{w}"]=df["close"].rolling(w).mean()

    df["close_vs_ma5"]=(df["close"]-df["_ma5"])/df["_ma5"]
    df["ma5_smooth_signal"]=(df["_ma5"]-df["_ma5"].shift(1))/df["close"].shift(1)
    df["momentum_5_20"]=df["ret_5d"]-df["close"].pct_change(20)

    log_ret=np.log(df["close"]/df["close"].shift(1))

    df["vol_20d"]=log_ret.rolling(20).std()
    df["vol_50d"]=log_ret.rolling(50).std()

    df["rsi_14"]=RSIIndicator(close=df["close"],window=14).rsi()

    volume_ma20=df["volume"].rolling(20).mean()
    df["volume_normalized"]=df["volume"]/volume_ma20

    df["dow"]=df["date"].dt.dayofweek

    rolling_mean=df["ret_1d"].rolling(20).mean()
    rolling_std=df["ret_1d"].rolling(20).std()

    df["ret_zscore"]=(df["ret_1d"]-rolling_mean)/rolling_std

    vix["vix_level"]=vix["close"]
    vix["vix_change"]=vix["close"].pct_change(1)

    df=df.merge(vix[["date","vix_level","vix_change"]],on="date",how="left")

    df["ema50"]=df["close"].ewm(span=50).mean()

    df["target"]=np.where(df["close"].pct_change(1).shift(-1)>0,1,0)

    ma_cols=[c for c in df.columns if c.startswith("_ma")]
    df.drop(columns=ma_cols,inplace=True)

    missing=[c for c in FEATURES if c not in df.columns]

    if len(missing)>0:
        raise ValueError(f"Missing generated features: {missing}")

    df.dropna(subset=FEATURES+["target"],inplace=True)
    df.reset_index(drop=True,inplace=True)

    return df

def compute_baselines(y):
    majority_class=int(np.mean(y)>=0.5)
    majority_acc=round(float(np.mean(y==majority_class)),4)
    persistence_acc=round(float(np.mean(y[1:]==y[:-1])),4)
    majority_label="up" if majority_class==1 else "down"

    return {
        "majority_class":majority_label,
        "majority_acc":majority_acc,
        "persistence_acc":persistence_acc
    }

def walk_forward(df_is):
    MIN_TRAIN=252
    STEP=63

    X=df_is[FEATURES].values
    y=df_is["target"].values

    rows=[]
    start=MIN_TRAIN

    while start+STEP<=len(df_is):
        Xtr=X[:start]
        ytr=y[:start]

        Xva=X[start:start+STEP]
        yva=y[start:start+STEP]

        sc=StandardScaler()

        Xtr_scaled=sc.fit_transform(Xtr)
        Xva_scaled=sc.transform(Xva)

        mdl=XGBClassifier(**XGB_PARAMS)

        mdl.fit(Xtr_scaled,ytr)

        preds=mdl.predict(Xva_scaled)
        proba=mdl.predict_proba(Xva_scaled)[:,1]

        rows.append({
            "fold":len(rows)+1,
            "train_end":str(df_is["date"].iloc[start-1].date()),
            "val_start":str(df_is["date"].iloc[start].date()),
            "val_end":str(df_is["date"].iloc[min(start+STEP-1,len(df_is)-1)].date()),
            "auc":round(roc_auc_score(yva,proba),4),
            "bal_acc":round(balanced_accuracy_score(yva,preds),4),
            "acc":round(accuracy_score(yva,preds),4)
        })

        start+=STEP

    return pd.DataFrame(rows)

def train_final(df_is):
    sc=StandardScaler()
    X_scaled=sc.fit_transform(df_is[FEATURES].values)

    mdl=XGBClassifier(**XGB_PARAMS)
    mdl.fit(X_scaled,df_is["target"].values)

    return sc,mdl

def evaluate_oos(df_oos,sc,mdl):
    Xoos=sc.transform(df_oos[FEATURES].values)
    yoos=df_oos["target"].values

    preds=mdl.predict(Xoos)
    proba=mdl.predict_proba(Xoos)[:,1]

    rng=np.random.default_rng(42)

    n=len(yoos)

    aucs=[]
    baccs=[]

    for _ in range(1000):
        idx=rng.integers(0,n,n)

        if len(np.unique(yoos[idx]))<2:
            continue

        aucs.append(roc_auc_score(yoos[idx],proba[idx]))
        baccs.append(balanced_accuracy_score(yoos[idx],preds[idx]))

    baselines=compute_baselines(yoos)

    metrics={
        "accuracy":round(float(accuracy_score(yoos,preds)),4),
        "precision":round(float(precision_score(yoos,preds)),4),
        "recall":round(float(recall_score(yoos,preds)),4),
        "f1":round(float(f1_score(yoos,preds)),4),
        "auc_mean":round(float(np.mean(aucs)),4),
        "auc_ci_lo":round(float(np.percentile(aucs,2.5)),4),
        "auc_ci_hi":round(float(np.percentile(aucs,97.5)),4),
        "bal_acc_mean":round(float(np.mean(baccs)),4),
        "bal_acc_ci_lo":round(float(np.percentile(baccs,2.5)),4),
        "bal_acc_ci_hi":round(float(np.percentile(baccs,97.5)),4),
        "confusion_matrix":confusion_matrix(yoos,preds).tolist(),
        "baseline_majority_class":baselines["majority_class"],
        "baseline_majority_acc":baselines["majority_acc"],
        "baseline_persistence_acc":baselines["persistence_acc"]
    }

    df_out=df_oos.copy()
    df_out["y_pred"]=preds
    df_out["y_proba"]=proba

    return metrics,df_out

def run_backtest(df_oos_pred):
    df_bt=df_oos_pred[["date","open","high","low","close","volume","ema50","y_proba"]].copy()

    df_bt=df_bt.rename(columns={
        "date":"Date",
        "open":"Open",
        "high":"High",
        "low":"Low",
        "close":"Close",
        "volume":"Volume"
    }).set_index("Date")

    class EmaXgbProb(Strategy):
        buy_threshold=BUY_THRESHOLD
        sell_threshold=SELL_THRESHOLD
        exit_bar=EXIT_BAR

        def init(self):
            self.ema=self.I(lambda x:x,self.data.ema50)
            self.proba=self.I(lambda x:x,self.data.y_proba)
            self.entry_bar=None
            self.signal_type=None

        def next(self):
            prob=self.proba[-1]

            above_ema=self.data.Close[-1]>self.ema[-1]
            below_ema=self.data.Close[-1]<self.ema[-1]

            in_position=bool(self.position)

            current_bar=len(self.data.Close)

            if in_position:
                bars_held=current_bar-self.entry_bar

                if self.signal_type=="long":
                    if bars_held>=self.exit_bar or prob<self.sell_threshold or below_ema:
                        self.position.close()
                        self.entry_bar=None
                        self.signal_type=None

                elif self.signal_type=="short":
                    if bars_held>=self.exit_bar or prob>self.buy_threshold or above_ema:
                        self.position.close()
                        self.entry_bar=None
                        self.signal_type=None

            else:
                if prob>self.buy_threshold and above_ema:
                    self.buy()
                    self.entry_bar=current_bar
                    self.signal_type="long"

                elif prob<self.sell_threshold and below_ema:
                    self.sell()
                    self.entry_bar=current_bar
                    self.signal_type="short"

    bt=Backtest(df_bt,EmaXgbProb,cash=10_000_000,commission=0.0002,trade_on_close=True)

    stats=bt.run()

    return {
        "buy_threshold":BUY_THRESHOLD,
        "sell_threshold":SELL_THRESHOLD,
        "exit_bar":EXIT_BAR,
        "sharpe":round(float(stats["Sharpe Ratio"]),4),
        "max_drawdown":round(float(stats["Max. Drawdown [%]"]),4),
        "hit_rate":round(float(stats["Win Rate [%]"]),4),
        "return_pct":round(float(stats["Return [%]"]),4),
        "num_trades":int(stats["# Trades"])
    }

if __name__=="__main__":
    print("1/5 Building features...")

    df=build_features()

    df_is=df[df["date"]<OOS_START].copy()
    df_oos=df[df["date"]>=OOS_START].copy()

    if len(df_oos)==0:
        raise ValueError(f"No OOS rows found after {OOS_START}")

    df.to_csv(os.path.join(ARTIFACTS_DIR,"features_final.csv"),index=False)

    print(f"IS={len(df_is)} | OOS={len(df_oos)}")
    print(f"OOS period: {df_oos['date'].min().date()} → {df_oos['date'].max().date()}")

    print("\n2/5 Walk-forward CV...")

    fold_df=walk_forward(df_is)

    fold_df.to_csv(os.path.join(ARTIFACTS_DIR,"wf_folds.csv"),index=False)

    print(fold_df[["fold","auc","bal_acc"]].to_string(index=False))

    print("\n3/5 Training final model...")

    sc,mdl=train_final(df_is)

    with open(os.path.join(ARTIFACTS_DIR,"scaler.pkl"),"wb") as f:
        pickle.dump(sc,f)

    with open(os.path.join(ARTIFACTS_DIR,"model.pkl"),"wb") as f:
        pickle.dump(mdl,f)

    print("\n4/5 OOS evaluation...")

    oos_m,df_oos_pred=evaluate_oos(df_oos,sc,mdl)

    with open(os.path.join(ARTIFACTS_DIR,"oos_metrics.json"),"w") as f:
        json.dump(oos_m,f,indent=2)

    df_oos_pred.to_csv(os.path.join(ARTIFACTS_DIR,"oos_predictions.csv"),index=False)

    print(json.dumps(oos_m,indent=2))

    print("\n5/5 Backtest...")

    bt_m=run_backtest(df_oos_pred)

    with open(os.path.join(ARTIFACTS_DIR,"bt_metrics.json"),"w") as f:
        json.dump(bt_m,f,indent=2)

    with open(os.path.join(ARTIFACTS_DIR,"xgb_params.json"),"w") as f:
        json.dump(XGB_PARAMS,f,indent=2)

    with open(os.path.join(ARTIFACTS_DIR,"features_list.json"),"w") as f:
        json.dump(FEATURES,f,indent=2)

    print(json.dumps(bt_m,indent=2))

    print(f"\nDone -> artifacts saved in:\n{ARTIFACTS_DIR}")