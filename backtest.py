import datetime

import pandas as pd
from backtesting import Backtest

from strategies.direction_confidence_strategy import DirectionConfidenceStrategy
from strategies.direction_model_strategy import DirectionModelStrategy
from strategies.duo_model_strategy import DuoModelStrategy
from strategies.label_test_strategy import LabelTestStrategy


def main():
    history = pd.read_pickle(
        "./data/processed/USDJPY-1m-20210101-20241231-processed.pkl"
    )

    labels = pd.read_pickle("./data/predictions/USDJPY-1m-20210101-20241231-CUSUM.pkl")

    history = history.join(labels, how="left")
    history["time"] = history.index
    history.set_index("time", inplace=True)
    history.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
            "side": "side",
            "bin": "bin",
        },
        inplace=True,
    )

    # Run backtest
    backtest = Backtest(
        history,
        # LabelTestStrategy,
        DirectionModelStrategy,
        cash=10000,
        margin=0.01,
        hedging=True,
        exclusive_orders=False,
    )
    result = backtest.run()

    print(result)
    print(f"Buy count = {result._strategy.buy_count}")
    print(f"Sell count = {result._strategy.sell_count}")


def log(msg):
    now = datetime.datetime.now()
    now_str = now.strftime("%Y.%m.%d %H:%M:%S")
    print(f"{now_str} {msg}")


if __name__ == "__main__":
    main()
