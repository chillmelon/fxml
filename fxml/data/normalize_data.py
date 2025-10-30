import os
import re
from pathlib import Path
from typing import Union

import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SCALER_MAP = {
    "std": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "log": None,
    "none": None,
}


def normalize(
    data: pd.DataFrame,
    data_name: str,
    scaler_cfg: DictConfig,
    scaler_dir="data/processed/scalers",
):
    """
    normalize dataframe

    Args:
        data: multi-column dataframe
        prefix: prefix for scaler path
        scaler_cfg: config for scaler

    Returns:
        normalized dataframe
    """
    # loop through columns
    for col, params in scaler_cfg.items():
        # skip if column not in dataframe
        if col not in data.columns:
            print(f"⚠️ Skip {col}: not in dataset")
            continue
        # load scaler class from file
        s_type = params["type"]
        if s_type == "none":
            continue
        elif s_type == "log":
            # revert log scaling
            data[col] = np.log1p(data[col])
        else:
            scaler = joblib.load(Path(f"{scaler_dir}/{data_name}/{col}_{s_type}.pkl"))
            data[col] = scaler.transform(data[[col]])
    return data


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing")
def main(config: DictConfig):
    train_data_path = Path(config.data.train_data_path)
    test_data_path = Path(config.data.test_data_path)
    train_df = pd.read_pickle(train_data_path)
    test_df = pd.read_pickle(test_data_path)

    save_dir = Path("data/processed/scalers") / Path(config.data.train_data_path).stem
    save_dir.mkdir(exist_ok=True)

    scaler_cfg = config.scaler

    print(f"✅ Loading scaler config: {scaler_cfg}")

    print(f"✅ Loaded scaler config ({len(scaler_cfg)} features)")

    for col, params in scaler_cfg.items():
        if col not in train_df.columns:
            print(f"⚠️ Skip {col}: not in dataset")
            continue

        s_type = params["type"]

        # ========== no scaling ==========
        if s_type == "none":
            continue

        # ========== log scaling ==========
        if s_type == "log":
            train_df[col] = np.log1p(train_df[col])
            test_df[col] = np.log1p(test_df[col])
            continue

        # ========== sklearn scaler ==========
        scaler_cls = SCALER_MAP.get(s_type)
        if scaler_cls is None:
            raise ValueError(f"❌ Unsupported scaler type: {s_type}")

        scaler = scaler_cls()
        scaler.fit(train_df[[col]])

        train_df[col] = scaler.transform(train_df[[col]])
        test_df[col] = scaler.transform(test_df[[col]])

        scaler_path = save_dir / f"{col}_{s_type}.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"    → Saved scaler: {scaler_path.name}")

    # 儲存 normalized 資料
    train_norm_path = train_data_path.with_name(train_data_path.stem + "_NORM.pkl")
    test_norm_path = test_data_path.with_name(test_data_path.stem + "_NORM.pkl")
    train_df.to_pickle(train_norm_path)
    test_df.to_pickle(test_norm_path)

    print(f"✅ Normalized data saved to:\n  {train_norm_path}\n  {test_norm_path}")

    return


def denorm(data, col_name, cfg):
    """
    反轉 normalization。
    Args:
        data: ndarray / Series
        col_name: 欄位名稱
        data_path: 原始 data 檔案路徑（用於推導 scaler 目錄）
        cfg: 讀入的 Hydra config（cfg.scaler）
    """
    s_type = cfg.scaler[col_name]["type"]
    scaler_dir = Path("data/processed/scalers") / Path(cfg.data.train_data_path).stem
    scaler_path = scaler_dir / f"{col_name}_{s_type}.pkl"

    if s_type == "none":
        return data
    elif s_type == "log":
        return np.expm1(data)
    elif not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    scaler = joblib.load(scaler_path)
    return scaler.inverse_transform(np.asarray(data).reshape(-1, 1)).ravel()


if __name__ == "__main__":
    main()
