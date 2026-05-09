"""feature_engineering.py file"""
import pandas as pd
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_intraday",
    "ret_overnight",
    "high_low_range",
    "volume_ratio_20d",
    "momentum_5_20",
    "vol_20d",
    "rsi_14",
    "bn_ret_1d",
    "vix_change",
]

def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    close   = df["close"]
    open_   = df["open"]
    high    = df["high"]
    low     = df["low"]
    volume  = df["volume"]
    bn_close = df["bn_close"]
    vix      = df["vix_close"]

    # ── returns ──────────────────────────────────────────────────────────────
    ret_1d  = close.pct_change()
    ret_5d  = close.pct_change(5)
    ret_20d = close.pct_change(20)

    # intraday: (close - open) / open  — fully known at EOD T, leak-safe
    ret_intraday = (close - open_) / (open_ + 1e-9)

    # overnight gap: today's open vs yesterday's close — known at market open T
    ret_overnight = (open_ - close.shift(1)) / (close.shift(1) + 1e-9)

    # ── range / volume ────────────────────────────────────────────────────────
    high_low_range  = (high - low) / (low + 1e-9)
    vol_ma20        = volume.rolling(20).mean()
    volume_ratio_20d = volume / (vol_ma20 + 1e-9)

    # ── momentum ─────────────────────────────────────────────────────────────
    momentum_5_20 = ret_5d - ret_20d

    # ── realised volatility ───────────────────────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    vol_20d = log_ret.rolling(20).std()

    # ── RSI-14 ───────────────────────────────────────────────────────────────
    gain  = close.diff().clip(lower=0)
    loss  = -close.diff().clip(upper=0)
    rs    = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
    rsi_14 = 100 - (100 / (1 + rs))

    # ── cross-asset ───────────────────────────────────────────────────────────
    bn_ret_1d = bn_close.pct_change()

    # ── VIX ──────────────────────────────────────────────────────────────────
    vix_change = vix.pct_change()

    # ── assign ───────────────────────────────────────────────────────────────
    df["ret_1d"]           = ret_1d
    df["ret_5d"]           = ret_5d
    df["ret_20d"]          = ret_20d
    df["ret_intraday"]     = ret_intraday
    df["ret_overnight"]    = ret_overnight
    df["high_low_range"]   = high_low_range
    df["volume_ratio_20d"] = volume_ratio_20d
    df["momentum_5_20"]    = momentum_5_20
    df["vol_20d"]          = vol_20d
    df["rsi_14"]           = rsi_14
    df["bn_ret_1d"]        = bn_ret_1d
    df["vix_change"]       = vix_change

    # ── target: next-day close direction ─────────────────────────────────────
    # shift(-1) looks one row ahead — safe because we dropna immediately after,
    # which removes the final row where shift(-1) would reach outside the data.
    df["target"] = (close.shift(-1) > close).astype(int)

    df = df.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)

    return df[FEATURES + ["date", "open", "high", "low", "close", "volume", "target"]]