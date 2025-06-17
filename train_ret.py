from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from datamodules.event_based_data_module import EventBasedDataModule
from models.classification.gru_model import GRUModule
from models.classification.t2v_transformer_model import T2VTransformerModule
from models.classification.transformer_model import TransformerModule

# Configurable parts
MODEL_NAME = "gru_regr"
SOURCE = "dukascopy"
SYMBOL = "usdjpy"
MINUTES = 1
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
EVENT_NAME = "cusum_filter"
SEQUENCE_LENGTH = 48
TIME_COLS = [
    # 'timestamp',
    # "hour",
    # "dow",
    # "dom",
    # "month",
    "open",
    "high",
    "low",
    "close",
]
FEATURES_COLS = [
    # Basic Data
    "close_log_return",
    "log_volume",
    "macd",
    "macd_signal",
    "macd_diff",
]
TARGET_COL = "ret"


# Build base name
BASE_NAME = f"{SOURCE}-{SYMBOL}-tick-{START_DATE}-{END_DATE}"
RESAMPLED_NAME = f"{SOURCE}-{SYMBOL}-{MINUTES}m-{START_DATE}-{END_DATE}"
LABEL_NAME = f"{RESAMPLED_NAME}-{EVENT_NAME}"
# Base directories
BASE_DIR = Path("./data")
RESAMPLED_DIR = BASE_DIR / "resampled"
LABEL_DIR = BASE_DIR / "labels"
PROCESSED_DIR = BASE_DIR / "processed"
DIRECTION_LABEL_DIR = BASE_DIR / "direction_labels"

# Final paths
PROCESSED_FILE_PATH = PROCESSED_DIR / f"{RESAMPLED_NAME}_processed.pkl"
DIRECTION_LABEL_FILE_PATH = DIRECTION_LABEL_DIR / f"{RESAMPLED_NAME}-{EVENT_NAME}.pkl"


def main():
    df = pd.read_pickle(PROCESSED_FILE_PATH)
    print(df.head())
    label_df = pd.read_pickle(DIRECTION_LABEL_FILE_PATH)

    print(df.loc[label_df.index].head())

    print(label_df[TARGET_COL].describe())

    dm = EventBasedDataModule(
        data=df,
        labels=label_df,
        sequence_length=SEQUENCE_LENGTH,
        features=FEATURES_COLS,
        target=TARGET_COL,
        batch_size=64,
    )
    dm.setup()

    # Initialize GRU module
    model = GRUModule(
        n_features=len(FEATURES_COLS),
        output_size=1,
        n_hidden=64,
        n_layers=2,
        dropout=0.6,
    )

    logger = TensorBoardLogger(
        "lightning_logs", name=f"{MODEL_NAME}-{MINUTES}-{EVENT_NAME}"
    )

    profiler = SimpleProfiler(filename="profiler")

    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=5, verbose=True
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
        # gradient_clip_val=1.0
        # num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
