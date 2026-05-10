"""backtest.py file"""
from backtesting import Backtest, Strategy
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json

import warnings
warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
ART      = ROOT / "artifacts"

model    = pickle.load(open(ART / "model.pkl",  "rb"))
scaler   = pickle.load(open(ART / "scaler.pkl", "rb"))
features = pd.read_json(ART / "features.json", typ="series").tolist()

THRESHOLD = 0.55

class S(Strategy):

    def init(self):
        self.ema50 = self.I(lambda x: x, self.data.ema50)

    def next(self):

        x = np.array([[getattr(self.data, f)[-1] for f in features]])
        x = scaler.transform(x)

        p = model.predict_proba(x)[0, 1]

        if p >= THRESHOLD:
            if not self.position:
                self.buy()
        else:
            if self.position:
                self.position.close()


def compute_sharpe(equity_curve, risk_free_rate=0.0):

    returns = equity_curve["Equity"].pct_change().dropna()

    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / 252)

    std = excess_returns.std()

    if std == 0 or np.isnan(std):
        return 0.0

    sharpe = (excess_returns.mean() / std) * np.sqrt(252)

    return float(sharpe)


def run():

    df = pd.read_csv(ART / "oos_predictions.csv")
    df.columns = df.columns.str.lower()

    from data_preprocessing import build_dataset
    from feature_engineering import create_features

    DATA_DIR = ROOT / "data" / "csv"

    full_df = create_features(build_dataset(DATA_DIR))

    ema50_full = (
        full_df
        .set_index("date")["close"]
        .ewm(span=50)
        .mean()
    )

    df["date"] = pd.to_datetime(df["date"])

    df["ema50"] = df["date"].map(ema50_full)

    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    df = df.dropna(
        subset=["Open", "High", "Low", "Close", "ema50"]
    ).reset_index(drop=True)

    bt = Backtest(
        df,
        S,
        cash=100_000,
        exclusive_orders=True
    )

    stats = bt.run()

    sharpe_manual = compute_sharpe(stats["_equity_curve"])

    stats["Custom Sharpe Ratio"] = sharpe_manual

    print(stats)

    stats_clean = {
        k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
        for k, v in stats.items()
        if k not in ("_equity_curve", "_trades", "_strategy")
    }

    with open(ART / "backtest_stats.json", "w") as f:
        json.dump(stats_clean, f, indent=2)

    bt.plot()


if __name__ == "__main__":
    run()