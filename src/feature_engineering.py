from pathlib import Path
import pandas as pd
import numpy as np
from logger import get_logger

logger=get_logger(__name__)

FEATURES=["ret_1d","ret_5d","ret_overnight","ret_zscore","close_vs_ma20","momentum_5_20","vol_20d","rsi_14","volume_ratio_20d","nifty_bn_spread","vix_change","close_vs_252d_high"]

def create_features(df):

 logger.info("Creating features")

 close=df["close"]
 open_=df["open"]
 high=df["high"]
 low=df["low"]
 volume=df["volume"]
 bn_close=df["bn_close"]

 log_ret=np.log(close/close.shift(1))
 bn_ret=bn_close.pct_change(1,fill_method=None)

 ma5=close.rolling(5).mean()
 ma20=close.rolling(20).mean()
 ma50=close.rolling(50).mean()

 vol_ma20=volume.rolling(20).mean()

 ret_1d=close.pct_change(1,fill_method=None)

 rolling_mean=ret_1d.rolling(20).mean()
 rolling_std=ret_1d.rolling(20).std()

 gain=close.diff().clip(lower=0)
 loss=close.diff().clip(upper=0).abs()

 rs=gain.rolling(14).mean()/(loss.rolling(14).mean()+1e-9)

 df["ret_1d"]=ret_1d
 df["ret_5d"]=close.pct_change(5,fill_method=None)
 df["ret_10d"]=close.pct_change(10,fill_method=None)
 df["ret_20d"]=close.pct_change(20,fill_method=None)

 df["ret_intraday"]=(close-open_)/open_
 df["ret_overnight"]=(open_-close.shift(1))/close.shift(1)

 df["high_low_range"]=(high-low)/low
 df["log_volume"]=np.log(volume.replace(0,np.nan))

 df["volume_ratio_20d"]=volume/(vol_ma20+1e-9)

 df["close_vs_ma5"]=(close-ma5)/ma5
 df["close_vs_ma20"]=(close-ma20)/ma20
 df["close_vs_ma50"]=(close-ma50)/ma50

 df["momentum_5_20"]=df["ret_5d"]-df["ret_20d"]

 df["vol_5d"]=log_ret.rolling(5).std()
 df["vol_20d"]=log_ret.rolling(20).std()
 df["vol_50d"]=log_ret.rolling(50).std()

 df["rsi_14"]=100-(100/(1+rs))

 df["bn_ret_1d"]=bn_ret
 df["bn_ret_5d"]=bn_close.pct_change(5,fill_method=None)

 df["nifty_bn_spread"]=ret_1d-bn_ret

 df["nifty_bn_corr_20d"]=ret_1d.rolling(20).corr(bn_ret)

 df["vix_level"]=df["vix_close"]
 df["vix_change"]=df["vix_close"].pct_change(1,fill_method=None)
 df["vix_5d_change"]=df["vix_close"].pct_change(5,fill_method=None)
 df["vix_ma_ratio"]=df["vix_close"]/(df["vix_close"].rolling(20).mean()+1e-9)

 df["close_vs_252d_high"]=(close-close.rolling(252).max())/(close.rolling(252).max()+1e-9)
 df["close_vs_252d_low"]=(close-close.rolling(252).min())/(close.rolling(252).min()+1e-9)

 df["dow"]=df["date"].dt.dayofweek

 df["ret_zscore"]=(ret_1d-rolling_mean)/(rolling_std+1e-9)

 df["ma5_smooth_signal"]=ma5.pct_change(1,fill_method=None)

 df["volume_normalized"]=volume/(vol_ma20+1e-9)

 df["ema50"]=close.ewm(span=50,adjust=False).mean()

 logger.info("Creating target")

 df["target"]=(df["close"].shift(-1)>df["close"]).astype(int)

 logger.info("Dropping NaNs")

 before=len(df)
 df=df.dropna(subset=FEATURES+["target"]).reset_index(drop=True)

 logger.info(f"Dropped {before-len(df)} rows")
 logger.info(f"Final shape: {df.shape}")

 return df[FEATURES+["date","open","high","low","close","volume","ema50","target"]]