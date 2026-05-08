from pathlib import Path
import pandas as pd
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

FEATURES = [
    "ret_1d",
    "ret_10d",
    "ret_intraday",
    "ret_overnight",
    "ret_zscore",
    "close_vs_ma50",
    "close_vs_252d_high",
    "rsi_14",
    "volume_ratio_20d",
    "vol_20d",
    "vix_change",
    "nifty_bn_spread"
]

def load_data(data_dir: Path):

    nifty = pd.read_csv(data_dir / "nifty50.csv")
    vix = pd.read_csv(data_dir / "indiavix.csv")
    banknifty = pd.read_csv(data_dir / "banknifty.csv")
    starter = pd.read_csv(data_dir / "starter_features.csv")

    for df in [nifty, vix, banknifty, starter]:
        df.columns = df.columns.str.strip().str.lower()
        df["date"] = pd.to_datetime(df["date"])

    nifty = nifty.sort_values("date").reset_index(drop=True)
    vix = vix.sort_values("date").reset_index(drop=True)
    banknifty = banknifty.sort_values("date").reset_index(drop=True)
    starter = starter.sort_values("date").reset_index(drop=True)

    return nifty, vix, banknifty, starter

def recompute_missing_features(df: pd.DataFrame,
                               vix: pd.DataFrame,
                               banknifty: pd.DataFrame):

    close = df["close"]
    open_ = df["open"]
    volume = df["volume"]

    log_ret = np.log(close / close.shift(1))

    ma50 = close.rolling(50).mean()

    rolling_mean = close.pct_change(1).rolling(20).mean()
    rolling_std = close.pct_change(1).rolling(20).std()

    rolling_high_252 = close.rolling(252).max()

    volume_ma20 = volume.rolling(20).mean()

    gain = close.diff().clip(lower=0)
    loss = close.diff().clip(upper=0).abs()

    rs = gain.rolling(14).mean() / loss.rolling(14).mean()

    banknifty["bn_ret_1d_calc"] = banknifty["close"].pct_change(1)

    bn_tmp = banknifty[["date", "bn_ret_1d_calc"]]

    vix["vix_change_calc"] = vix["close"].pct_change(1)

    vix_tmp = vix[["date", "close", "vix_change_calc"]]
    vix_tmp = vix_tmp.rename(columns={"close": "vix_level_calc"})

    df = df.merge(vix_tmp, on="date", how="left")
    df = df.merge(bn_tmp, on="date", how="left")

    calculated = {
        "ret_1d": close.pct_change(1),
        "ret_10d": close.pct_change(10),
        "ret_intraday": (close - open_) / open_,
        "ret_overnight": (open_ - close.shift(1)) / close.shift(1),
        "ret_zscore": (close.pct_change(1) - rolling_mean) / rolling_std,
        "close_vs_ma50": (close - ma50) / ma50,
        "close_vs_252d_high": (close - rolling_high_252) / rolling_high_252,
        "rsi_14": 100 - (100 / (1 + rs)),
        "volume_ratio_20d": volume / volume_ma20,
        "vol_20d": log_ret.rolling(20).std(),
        "vix_change": df["vix_change_calc"],
        "nifty_bn_spread": close.pct_change(1) - df["bn_ret_1d_calc"]
    }

    for col, values in calculated.items():

        if col not in df.columns:
            df[col] = values

        else:
            mask = df[col].isna()
            df.loc[mask, col] = values[mask]

    if "vix_level" not in df.columns:
        df["vix_level"] = df["vix_level_calc"]

    else:
        mask = df["vix_level"].isna()
        df.loc[mask, "vix_level"] = df.loc[mask, "vix_level_calc"]

    if "bn_ret_1d" not in df.columns:
        df["bn_ret_1d"] = df["bn_ret_1d_calc"]

    else:
        mask = df["bn_ret_1d"].isna()
        df.loc[mask, "bn_ret_1d"] = df.loc[mask, "bn_ret_1d_calc"]

    df = df.drop(columns=[
        "vix_level_calc",
        "vix_change_calc",
        "bn_ret_1d_calc"
    ], errors="ignore")

    return df

def build_dataset(data_dir: Path):

    logger.info("Loading datasets")

    nifty, vix, banknifty, starter = load_data(data_dir)

    logger.info("Preparing starter features")

    starter_cols = ["date"] + [c for c in FEATURES if c in starter.columns]

    extra_cols = [
        "vix_level",
        "bn_ret_1d"
    ]

    for col in extra_cols:
        if col in starter.columns and col not in starter_cols:
            starter_cols.append(col)

    starter = starter[starter_cols]

    logger.info("Merging datasets")

    df = nifty.merge(starter, on="date", how="left")

    logger.info("Recomputing only missing feature values")

    df = recompute_missing_features(df, vix, banknifty)

    logger.info("Creating EMA50")

    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    logger.info("Creating target")

    df["target"] = np.where(
        df["close"].pct_change(1).shift(-1) > 0,
        1,
        0
    )

    logger.info("Dropping remaining NaNs")

    before = len(df)

    df = df.dropna(subset=FEATURES + ["target"])

    after = len(df)

    logger.info(f"Dropped {before - after} rows")

    df = df.reset_index(drop=True)

    logger.info(f"Final dataset shape: {df.shape}")

    logger.info(f"Final features used: {FEATURES}")

    return df