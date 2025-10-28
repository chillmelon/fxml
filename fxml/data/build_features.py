from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from fxml.data.preprocessing.features import (
    add_returns,
    add_technical_indicators,
    add_time_features,
)


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing")
def main(config: DictConfig):
    symbol = config.get("symbol", {}).get("symbol", "USDJPY")
    minutes = config.get("minutes", [5])
    date_ranges = config.get("date", [])

    for date_range in date_ranges:
        date_from = str(date_range.get("from", None))
        date_to = str(date_range.get("to", None))
        for minute in minutes:
            RESAMPLED_NAME = f"{symbol}-{minute}m-{date_from}-{date_to}"

            resampled_path = f"data/resampled/{RESAMPLED_NAME}.pkl"

            df = pd.read_pickle(resampled_path)
            df.reset_index(inplace=True)

            df["log_volume"] = np.log1p(df["volume"])
            # add returns
            return_config = config.get("features", {}).get("returns", {})
            df = add_returns(df, config=return_config)

            # add TA
            ta_config = config.get("features", {}).get("technical_indicators", {})
            df = add_technical_indicators(df, config=ta_config)

            # add time features
            df = add_time_features(df)

            # drop NaN
            df.dropna(inplace=True)

            # set timestamp index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")

            # save file
            df.to_pickle(f"data/processed/{RESAMPLED_NAME}_FEATURES.pkl")

    return


if __name__ == "__main__":
    main()
