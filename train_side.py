from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from datamodules.event_based_data_module import EventBasedDataModule
from models.classification.gru_model import GRUModule
from models.classification.t2v_transformer_model import T2VTransformerModule
from models.classification.transformer_model import TransformerModule

# Configurable parts
MODEL_NAME = "t2v+transformer"
SOURCE = "dukascopy"
SYMBOL = "usdjpy"
EVENT = "58m-dollar"
MINUTES = 1
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
EVENT_NAME = "cusum_filter"
SEQUENCE_LENGTH = 120
TIME_COLS = [
    # "timestamp",
    "hour",
    "dow",
    "dom",
    "month",
    # "open",
    # "high",
    # "low",
    # "close",
]
FEATURES_COLS = [
    # Basic Data
    "close_log_return",
    "ret_mean_5",
    "ret_mean_10",
    "log_volume",
    "ema5_slope",
    "ema20_slope",
    "atr20",
    "vol_adj_return",
    "close_to_atr",
    "macd_diff",
    "bb_width",
    "bb_position",
    "donchian_width",
]
TARGET_COL = "bin_class"


# Build base name
BASE_NAME = f"{SOURCE}-{SYMBOL}-tick-{START_DATE}-{END_DATE}"
RESAMPLED_NAME = f"{SOURCE}-{SYMBOL}-{MINUTES}m-{START_DATE}-{END_DATE}"
RESAMPLED_NAME = f"{SOURCE}-{SYMBOL}-{EVENT}-{START_DATE}-{END_DATE}"
LABEL_NAME = f"{RESAMPLED_NAME}-{EVENT_NAME}"
# Base directories
BASE_DIR = Path("./data")
RESAMPLED_DIR = BASE_DIR / "resampled"
LABEL_DIR = BASE_DIR / "labels"
PROCESSED_DIR = BASE_DIR / "processed"
NORMALIZED_DIR = BASE_DIR / "normalized"
DIRECTION_LABEL_DIR = BASE_DIR / "direction_labels"

# Final paths
RESAMPLED_FILE_PATH = RESAMPLED_DIR / f"{RESAMPLED_NAME}.pkl"
PROCESSED_FILE_PATH = PROCESSED_DIR / f"{RESAMPLED_NAME}_processed.pkl"
NORMALIZED_FILE_PATH = NORMALIZED_DIR / f"{RESAMPLED_NAME}_normalized.pkl"
DIRECTION_LABEL_FILE_PATH = DIRECTION_LABEL_DIR / f"{RESAMPLED_NAME}-{EVENT_NAME}.pkl"


def main():
    df = pd.read_pickle(NORMALIZED_FILE_PATH)
    label_df = pd.read_pickle(DIRECTION_LABEL_FILE_PATH)

    print(df.loc[label_df.index].head())

    print(label_df[TARGET_COL].value_counts())
    X_COLS = TIME_COLS + FEATURES_COLS

    dm = EventBasedDataModule(
        data=df,
        labels=label_df,
        sequence_length=SEQUENCE_LENGTH,
        features=X_COLS,
        target=TARGET_COL,
        batch_size=256,
    )
    dm.setup()

    # Initialize GRU module
    # model = GRUModule(
    # n_features=len(FEATURES_COLS),
    # output_size=3,
    # n_hidden=64,
    # n_layers=2,
    # dropout=0.6,
    # )

    # model = TransformerModule(
    # n_features=len(FEATURES_COLS),
    # output_size=3,
    # num_layers=2,
    # d_model=64,
    # nhead=4,
    # dim_feedforward=256,
    # dropout=0.3,
    # )

    model = T2VTransformerModule(
        n_time=len(TIME_COLS),
        n_features=len(FEATURES_COLS),
        output_size=3,
        num_layers=3,
        d_model=128,
        nhead=4,
        dim_feedforward=256,
        kernel_size=6,
        dropout=0.4,
        label_smoothing=0.02,
    )

    logger = TensorBoardLogger(
        "lightning_logs", name=f"{MODEL_NAME}-{EVENT}-{EVENT_NAME}"
    )

    profiler = SimpleProfiler(filename="profiler")

    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=12, verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        filename="best_checkpoint",
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    # Training
    trainer = Trainer(
        accelerator="mps",
        devices=1,
        profiler=profiler,
        callbacks=[checkpoint_callback, early_stopping],
        max_epochs=100,
        logger=logger,
        # gradient_clip_val=1.0,
        # num_sanity_val_steps=0,
    )
    torch.set_float32_matmul_precision("high")
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
