import datetime
from pathlib import Path

import hydra
import pandas as pd
from backtesting import Backtest
from omegaconf import DictConfig

from fxml.trading.strategies.direction_confidence_strategy import (
    DirectionConfidenceStrategy,
)
from fxml.trading.strategies.direction_model_strategy import DirectionModelStrategy
from fxml.trading.strategies.duo_model_strategy import DuoModelStrategy
from fxml.trading.strategies.emacross_strategy import EmacrossStrategy
from fxml.trading.strategies.label_test_strategy import LabelTestStrategy


@hydra.main(version_base=None, config_path="./configs", config_name="trade")
def main(config: DictConfig):
    history = pd.read_pickle(config["data"]["dataset_path"])

    predictions = pd.read_pickle(
        Path("./data/predictions")
        / f"{config['model']['name']}_{Path(config["data"]["label_path"]).stem}.pkl"
    )

    history = history.join(predictions, how="left")
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
        EmacrossStrategy,
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
