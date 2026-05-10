"""feature_engineering.py file"""
import pandas as pd
import numpy as np
from logger import get_logger
import warnings
warnings.filterwarnings("ignore")

logger = get_logger(__name__)

FEATURES = [
    "ret_1d", "ret_zscore", "ma5_smooth_signal", "ret_intraday",
    "ret_overnight", "high_low_range", "volume_ratio_20d", "momentum_5_20",
    "vol_20d", "rsi_14", "bn_ret_1d", "vix_change",
]

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, open_, high, low = df["close"], df["open"], df["high"], df["low"]
    volume, bn_close, vix   = df["volume"], df["bn_close"], df["vix_close"]

    ret_1d  = close.pct_change()
    ret_5d  = close.pct_change(5)
    ret_20d = close.pct_change(20)
    log_ret = np.log(close / close.shift(1))
    ma5     = close.rolling(5).mean()

    gain = close.diff().clip(lower=0)
    loss = -close.diff().clip(upper=0)
    rs   = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)

    df["ret_1d"]           = ret_1d
    df["ret_zscore"]       = (ret_1d - ret_1d.rolling(20).mean()) / (ret_1d.rolling(20).std() + 1e-9)
    df["ma5_smooth_signal"]= (ma5 - ma5.shift(1)) / (close.shift(1) + 1e-9)
    df["ret_intraday"]     = (close - open_) / (open_ + 1e-9)
    df["ret_overnight"]    = (open_ - close.shift(1)) / (close.shift(1) + 1e-9)
    df["high_low_range"]   = (high - low) / (low + 1e-9)
    df["volume_ratio_20d"] = volume / (volume.rolling(20).mean() + 1e-9)
    df["momentum_5_20"]    = ret_5d - ret_20d
    df["vol_20d"]          = log_ret.rolling(20).std()
    df["rsi_14"]           = 100 - (100 / (1 + rs))
    df["bn_ret_1d"]        = bn_close.pct_change()
    df["vix_change"]       = vix.pct_change()
    df["target"]           = (close.shift(-1) > close).astype(int)

    df = df.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)
    return df[FEATURES + ["date", "open", "high", "low", "close", "volume", "target"]]