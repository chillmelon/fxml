from pathlib import Path

import pandas as pd

from fxml.data.preprocessing.features import (
    add_returns,
    add_technical_indicators,
    add_time_features,
)
from fxml.utils import load_config


def main():
    config = load_config("configs/features.yaml")
    resampled_path = config.get("data", {}).get("resampled", {})
    resampled_name = Path(resampled_path).stem

    df = pd.read_pickle(resampled_path)

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
    df.to_pickle(f"data/processed/{resampled_name}_FEATURES.pkl")

    return


if __name__ == "__main__":
    main()
