from pathlib import Path
import pandas as pd
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

FEATURES = [
    "ret_zscore","ret_1d","ret_5d","ret_10d","ret_20d",
    "ret_overnight","high_low_range","volume_ratio_20d",
    "momentum_5_20","vol_5d","vol_20d","vol_50d","rsi_14",
    "bn_ret_1d","bn_ret_5d","nifty_bn_spread",
    "vix_change","vix_5d_change","vix_ma_ratio"
]

def create_features(df):

    close, open_, high, low, volume, bn_close = df["close"], df["open"], df["high"], df["low"], df["volume"], df["bn_close"]

    ret_1d = close.pct_change()
    ret_5d = close.pct_change(5)
    ret_10d = close.pct_change(10)
    ret_20d = close.pct_change(20)

    log_ret = np.log(close / close.shift(1))

    ma20 = close.rolling(20).mean()
    vol_ma20 = volume.rolling(20).mean()

    bn_ret_1d = bn_close.pct_change()
    bn_ret_5d = bn_close.pct_change(5)

    gain = close.diff().clip(lower=0)
    loss = -close.diff().clip(upper=0)

    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
    rsi_14 = 100 - (100 / (1 + rs))

    rolling_mean = ret_1d.rolling(20).mean().shift(1)
    rolling_std = ret_1d.rolling(20).std().shift(1)
    ret_zscore = (ret_1d - rolling_mean) / (rolling_std + 1e-9)

    df["ret_1d"] = ret_1d
    df["ret_5d"] = ret_5d
    df["ret_10d"] = ret_10d
    df["ret_20d"] = ret_20d

    df["ret_overnight"] = (open_ - close.shift(1)) / (close.shift(1) + 1e-9)

    df["high_low_range"] = (high - low) / (low + 1e-9)

    df["volume_ratio_20d"] = volume / (vol_ma20 + 1e-9)

    df["momentum_5_20"] = ret_5d - ret_20d

    df["vol_5d"] = log_ret.rolling(5).std()
    df["vol_20d"] = log_ret.rolling(20).std()
    df["vol_50d"] = log_ret.rolling(50).std()

    df["rsi_14"] = rsi_14

    df["bn_ret_1d"] = bn_ret_1d
    df["bn_ret_5d"] = bn_ret_5d

    df["nifty_bn_spread"] = ret_1d - bn_ret_1d

    vix = df["vix_close"]
    df["vix_change"] = vix.pct_change()
    df["vix_5d_change"] = vix.pct_change(5)
    df["vix_ma_ratio"] = vix / (vix.rolling(20).mean() + 1e-9)

    df["ret_zscore"] = ret_zscore

    df["target"] = (close.shift(-1) > close).astype(int)

    df = df.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)

    return df[FEATURES + ["date","open","high","low","close","volume","target"]]