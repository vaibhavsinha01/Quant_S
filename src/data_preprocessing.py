from pathlib import Path
import pandas as pd
from logger import get_logger

logger = get_logger(__name__)

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def build_dataset(data_dir: Path):
    logger.info("Loading datasets")

    nifty = load_csv(data_dir / "nifty50.csv")
    vix = load_csv(data_dir / "indiavix.csv")
    banknifty = load_csv(data_dir / "banknifty.csv")

    logger.info("Preparing VIX")

    vix = vix.rename(columns={"close": "vix_close"})[["date", "vix_close"]]

    logger.info("Preparing BankNifty")

    banknifty = banknifty.rename(columns={"open": "bn_open", "high": "bn_high", "low": "bn_low", "close": "bn_close", "volume": "bn_volume"})[["date", "bn_open", "bn_high", "bn_low", "bn_close", "bn_volume"]]

    logger.info("Merging datasets")

    df = nifty.merge(vix, on="date", how="left").merge(banknifty, on="date", how="left")

    logger.info(f"Final shape: {df.shape}")

    return df