"""data_preprocessing.py file"""
from pathlib import Path
import pandas as pd
from logger import get_logger

logger = get_logger(__name__)

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"])
    return df.sort_values("date").reset_index(drop=True)

def build_dataset(data_dir: Path):

    nifty = load_csv(data_dir / "nifty50.csv")
    vix = load_csv(data_dir / "indiavix.csv")
    banknifty = load_csv(data_dir / "banknifty.csv")

    vix = vix.rename(columns={"close": "vix_close"})[["date", "vix_close"]]

    banknifty = banknifty.rename(columns={
        "open": "bn_open",
        "high": "bn_high",
        "low": "bn_low",
        "close": "bn_close",
        "volume": "bn_volume"
    })[["date", "bn_open", "bn_high", "bn_low", "bn_close", "bn_volume"]]

    df = nifty.merge(vix, on="date", how="left").merge(banknifty, on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)
    df = df.ffill().bfill()

    df = df.drop_duplicates(subset=["date"])

    logger.info(f"Final shape: {df.shape}")

    return df