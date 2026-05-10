"""backtest.py file"""
from backtesting import Backtest, Strategy
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
ART      = ROOT / "artifacts"

model    = pickle.load(open(ART / "model.pkl",  "rb"))
scaler   = pickle.load(open(ART / "scaler.pkl", "rb"))
features = pd.read_json(ART / "features.json", typ="series").tolist()

THRESHOLD = 0.55      # must match train_model.py and walkforward.py

class S(Strategy):

    def init(self):
        # EMA is passed in as a precomputed column so it has full history
        # from the complete price series, not just the OOS slice.
        # oos_predictions.csv carries an "ema50" column written by run() below.
        self.ema50 = self.I(lambda x: x, self.data.ema50)

    def next(self):
        # pull the latest feature row and scale it
        x = np.array([[getattr(self.data, f)[-1] for f in features]])
        x = scaler.transform(x)
        p = model.predict_proba(x)[0, 1]

        # long-only: buy on high-confidence up signal, flat otherwise.
        # no short positions — shorting NIFTY futures requires separate
        # margin treatment and is outside the scope of this backtest.
        if p >= THRESHOLD:
            if not self.position:
                self.buy()
        else:
            if self.position:
                self.position.close()

def run():
    # ── load OOS predictions ──────────────────────────────────────────────────
    df = pd.read_csv(ART / "oos_predictions.csv")
    df.columns = df.columns.str.lower()

    # ── precompute EMA-50 on the full Nifty price series so the OOS window
    #    has a correctly initialised EMA, not one that restarts from row 0 ────
    from data_preprocessing import build_dataset
    from feature_engineering import create_features, FEATURES
    DATA_DIR = ROOT / "data" / "csv"
    full_df  = create_features(build_dataset(DATA_DIR))
    ema50_full = full_df.set_index("date")["close"].ewm(span=50).mean()

    df["date"] = pd.to_datetime(df["date"])
    df["ema50"] = df["date"].map(ema50_full)

    # ── backtesting.py requires title-cased OHLCV columns ────────────────────
    df = df.rename(columns={
        "open":  "Open",
        "high":  "High",
        "low":   "Low",
        "close": "Close",
        "volume": "Volume",
    })

    df = df.dropna(subset=["Open", "High", "Low", "Close", "ema50"]).reset_index(drop=True)

    # ── run ───────────────────────────────────────────────────────────────────
    bt    = Backtest(df, S, cash=100_000, exclusive_orders=True)
    stats = bt.run()

    print(stats)

    # save backtest stats for the report
    stats_clean = {
        k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
        for k, v in stats.items()
        if k not in ("_equity_curve", "_trades", "_strategy")
    }
    import json
    with open(ART / "backtest_stats.json", "w") as f:
        json.dump(stats_clean, f, indent=2)

    bt.plot()

if __name__ == "__main__":
    run()