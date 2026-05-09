from backtesting import Backtest, Strategy
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

model = pickle.load(open(ART / "model.pkl", "rb"))
scaler = pickle.load(open(ART / "scaler.pkl", "rb"))
features = pd.read_json(ART / "features.json", typ="series").tolist()

class S(Strategy):
    def init(self):
        self.ema50 = self.I(lambda x: pd.Series(x).ewm(span=50).mean(), self.data.Close)

    def next(self):
        x = np.array([[getattr(self.data, f)[-1] for f in features]])
        x = scaler.transform(x)
        p = model.predict_proba(x)[0, 1]

        c = self.data.Close[-1]
        trend = c > self.ema50[-1]

        if p > 0.5 and trend:
            if not self.position:
                self.buy()
        elif p < 0.5 and not trend:
            if not self.position:
                self.sell()
        else:
            if self.position:
                self.position.close()

def run():
    df = pd.read_csv(ART / "oos_predictions.csv")
    df.columns = df.columns.str.lower()

    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)

    bt = Backtest(df, S, cash=100000, exclusive_orders=True)
    stats = bt.run()

    print(stats)
    bt.plot()

if __name__ == "__main__":
    run()