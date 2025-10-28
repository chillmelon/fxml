import os
import re
from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def categorize_column(col_name: str) -> str:
    regex_mapping = {
        "robust": [r"(volume)", r"(tick)"],
        "standard": [
            r"(spread)" r"(log_return)",
            r"(ema|_slope)",
            r"(bbb_|dc|macd)",
        ],
        "minmax": [
            r"\b(open|high|low|close)\b",
            r"(atr)",
            r"(adx|plus_di|minus_di)",
            r"(rsi)",
        ],
    }
    """Return which scaler a given column should use."""
    for scaler_type, patterns in regex_mapping.items():
        if any(re.search(pattern, col_name, re.I) for pattern in patterns):
            return scaler_type
    return "none"


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing")
def main(config: DictConfig):
    train_data_path = Path(config.data.train_data_path)
    test_data_path = Path(config.data.test_data_path)
    train_data = pd.read_pickle(train_data_path)
    test_data = pd.read_pickle(test_data_path)

    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()

    cols_by_scaler = {"robust": [], "standard": [], "minmax": [], "none": []}
    for col in numeric_cols:
        scaler_type = categorize_column(col)
        cols_by_scaler[scaler_type].append(col)

    scalers = {
        "robust": RobustScaler(),
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
    }

    for scaler_type, columns in cols_by_scaler.items():
        if columns and scaler_type != "none":
            scaler = scalers[scaler_type]
            train_data[columns] = scaler.fit_transform(train_data[columns])
            test_data[columns] = scaler.transform(test_data[columns])

            # Save scaler if directory provided
            scaler_path = (
                Path("data/processed/scalers")
                / f"{train_data_path.stem}_{scaler_type}_scaler.pkl"
            )
            joblib.dump(scaler, scaler_path)
            print(f"    → Saved {scaler_type} scaler to: {scaler_path}")
    train_data.to_pickle(Path(f"data/normalized/{train_data_path.stem}.pkl"))
    test_data.to_pickle(Path(f"data/normalized/{test_data_path.stem}.pkl"))

    return


if __name__ == "__main__":
    main()
