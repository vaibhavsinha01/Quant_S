from pathlib import Path
import pandas as pd
import numpy as np
import pickle,mlflow
from logger import get_logger

logger=get_logger(__name__)

PROJECT_ROOT=Path(__file__).resolve().parent.parent
ARTIFACTS_DIR=PROJECT_ROOT/"artifacts"

MODEL_PATH=ARTIFACTS_DIR/"model.pkl"
SCALER_PATH=ARTIFACTS_DIR/"scaler.pkl"
FEATURES_PATH=ARTIFACTS_DIR/"features.json"

MLFLOW_DB=PROJECT_ROOT/"mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")

def load_artifacts():
 model=pickle.load(open(MODEL_PATH,"rb"))
 scaler=pickle.load(open(SCALER_PATH,"rb"))
 features=pd.read_json(FEATURES_PATH,typ="series").tolist()
 return model,scaler,features

def run_backtest(df,model,scaler,features):

 df=df.sort_values("date").reset_index(drop=True)

 X=scaler.transform(df[features])
 proba=model.predict_proba(X)[:,1]

 df["proba"]=proba

 df["signal"]=0
 df.loc[df["proba"]>0.55,"signal"]=1
 df.loc[df["proba"]<0.45,"signal"]=-1

 df["ret"]=df["close"].pct_change().shift(-1)

 df["strategy_ret"]=df["signal"]*df["ret"]

 df["equity"]=(1+df["strategy_ret"].fillna(0)).cumprod()

 equity_final=df["equity"].iloc[-1]
 return_pct=(equity_final-1)*100

 trades=(df["signal"]!=0).sum()

 win_rate=(df[df["strategy_ret"]>0]["strategy_ret"].count()/max(trades,1))*100

 sharpe=df["strategy_ret"].mean()/df["strategy_ret"].std()*np.sqrt(252)

 stats={
 "equity_final":float(equity_final),
 "return_pct":float(return_pct),
 "trades":int(trades),
 "win_rate":float(win_rate),
 "sharpe":float(sharpe)
 }

 return df,stats

def main():

 logger.info("Loading artifacts")
 model,scaler,features=load_artifacts()

 data_path=PROJECT_ROOT/"artifacts"/"oos_predictions.csv"

 df=pd.read_csv(data_path)

 logger.info("Running backtest")

 result_df,stats=run_backtest(df,model,scaler,features)

 logger.info("Backtest completed")

 logger.info(f"Return %: {stats['return_pct']:.2f}")
 logger.info(f"Sharpe   : {stats['sharpe']:.2f}")
 logger.info(f"Trades   : {stats['trades']}")
 logger.info(f"Win rate : {stats['win_rate']:.2f}")

 result_df.to_csv(ARTIFACTS_DIR/"backtest_results.csv",index=False)

 print("\nRun MLflow UI with:")
 print(f"mlflow ui --backend-store-uri sqlite:///{MLFLOW_DB.as_posix()}")

if __name__=="__main__":
 main()