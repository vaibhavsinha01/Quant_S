from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from backtesting import Backtest, Strategy

from logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "features.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "oos_predictions.csv"

BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45
EXIT_BARS = 2

logger.info("Loading artifacts")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(FEATURES_PATH, "r") as f:
    FEATURES = json.load(f)

df = pd.read_csv(PREDICTIONS_PATH)

df.columns = df.columns.str.strip().str.lower()

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

required_cols = ["date", "close", "ema50", "probability"]

for col in required_cols:

    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df = df.rename(columns={
    "close": "Close"
})

df["Open"] = df["Close"]
df["High"] = df["Close"]
df["Low"] = df["Close"]
df["Volume"] = 1

df = df.set_index("date")

class MLStrategy(Strategy):

    def init(self):

        self.entry_bar = None

    def next(self):

        prob = self.data.probability[-1]
        close = self.data.Close[-1]
        ema50 = self.data.ema50[-1]

        current_bar = len(self.data)

        if self.position:

            if current_bar - self.entry_bar >= EXIT_BARS:
                self.position.close()

            return

        if prob >= BUY_THRESHOLD and close > ema50:

            self.buy()

            self.entry_bar = current_bar

        elif prob <= SELL_THRESHOLD and close < ema50:

            self.sell()

            self.entry_bar = current_bar

bt = Backtest(
    df,
    MLStrategy,
    cash=100000,
    commission=0.0005,
    exclusive_orders=True
)

logger.info("Running backtest")

stats = bt.run()

logger.info("Backtest completed")

print(stats)

bt.plot()