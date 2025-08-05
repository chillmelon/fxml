import datetime

import pandas as pd
from backtesting import Backtest

from strategies.label_test_strategy import LabelTestStrategy


def main():
    history = pd.read_pickle(
        "./data/normalized/dukascopy-usdjpy-58m-dollar-2020-01-01-2024-12-31_normalized.pkl"
    )

    labels = pd.read_pickle(
        "./data/direction_labels/dukascopy-usdjpy-58m-dollar-2020-01-01-2024-12-31-cusum_filter.pkl"
    )

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
            "bin": "side",
        },
        inplace=True,
    )

    # Run backtest
    test = Backtest(
        history,
        LabelTestStrategy,
        cash=10000,
        margin=0.01,
        hedging=True,
        exclusive_orders=False,
    )
    result = test.run()

    print(result)
    print(f"Buy count = {result._strategy.buy_count}")
    print(f"Sell count = {result._strategy.sell_count}")


def log(msg):
    now = datetime.datetime.now()
    now_str = now.strftime("%Y.%m.%d %H:%M:%S")
    print(f"{now_str} {msg}")


if __name__ == "__main__":
    main()
