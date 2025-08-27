# Copyright 2019-2025, Orchard Forex
# https://orchardforex.com

import datetime

import MetaTrader5 as mt5
import pandas as pd
from backtesting import Backtest

from strategies.gru_strategy import GRUStrategy
from strategies.simple_strategy import SimpleStrategy


def main():
    # Initialize MetaTrader 5
    if not mt5.initialize():
        log("Terminal initialization failed")
        return
    log("MT5 successfully initialized")

    # Download historical OHLCV data
    symbol = "USDJPY"
    timeframe = mt5.TIMEFRAME_H1
    bars = 10000

    history = pd.DataFrame(mt5.copy_rates_from_pos(symbol, timeframe, 0, bars))
    if history.empty:
        log(f"No data fetched for {symbol}")
        mt5.shutdown()
        return

    # Format the DataFrame for backtesting.py
    history["time"] = pd.to_datetime(history["time"], unit="s")
    history.set_index("time", inplace=True)
    history.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
        },
        inplace=True,
    )

    print(history.head())

    # Run backtest
    test = Backtest(
        history, GRUStrategy, cash=10000, hedging=True, exclusive_orders=True
    )
    result = test.run()

    print(result)
    print(f"Buy count = {result._strategy.buy_count}")
    print(f"Sell count = {result._strategy.sell_count}")

    mt5.shutdown()


def log(msg):
    now = datetime.datetime.now()
    now_str = now.strftime("%Y.%m.%d %H:%M:%S")
    print(f"{now_str} {msg}")


if __name__ == "__main__":
    main()
