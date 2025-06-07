from pathlib import Path
import pandas as pd
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
import torch

from datamodules.event_based_data_module import EventBasedDataModule
from models.transformer_model import TransformerModule
from models.transformer_model import TransformerModule


# Configurable parts
SOURCE = "dukascopy"
SYMBOL = "usdjpy"
MINUTES = 1
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
EVENT_NAME = 'cusum_filter'
SIDE_NAME = 'trend_line_same'
SEQUENCE_LENGTH=24
FEATURES_COLS = [
    # Basic Data
    'close_log_return',
    'log_volume',
]

TARGET_COL = 'ret'

# Build base name
BASE_NAME = f"{SOURCE}-{SYMBOL}-tick-{START_DATE}-{END_DATE}"
RESAMPLED_NAME = f"{SOURCE}-{SYMBOL}-{MINUTES}m-{START_DATE}-{END_DATE}"
LABEL_NAME = f"{RESAMPLED_NAME}-{EVENT_NAME}-{SIDE_NAME}"
# Base directories
BASE_DIR = Path("./data")
RESAMPLED_DIR = BASE_DIR / "resampled"
LABEL_DIR = BASE_DIR / "labels"

# Final paths
RESAMPLED_FILE_PATH = RESAMPLED_DIR / f"{RESAMPLED_NAME}.pkl"
LABEL_FILE_PATH = LABEL_DIR / f"{LABEL_NAME}.pkl"


def main():
    df = pd.read_pickle(RESAMPLED_FILE_PATH)
    label_df = pd.read_pickle(LABEL_FILE_PATH)

    dm = EventBasedDataModule(
        data=df,
        labels=label_df,
        sequence_length=SEQUENCE_LENGTH,
        features=FEATURES_COLS,
        target=TARGET_COL,
    )
    dm.setup()

    model = TransformerModule(
        n_features=len(FEATURES_COLS),
        output_size=3,
        num_layers=3,
        d_model=64,
        nhead=4,
        dim_feedforward=256,
        dropout=0.3
    )

    logger = TensorBoardLogger("lightning_logs", name="transformer_m15_3barrier")

    profiler = SimpleProfiler(filename='profiler')

    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        filename='best_checkpoint',
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Setup tensor core
    torch.set_float32_matmul_precision('high')
    # Training
    trainer = Trainer(
        accelerator="gpu",
        precision='16-mixed',
        profiler=profiler,
        callbacks=[checkpoint_callback, early_stopping],
        max_epochs=100,
        logger=logger,
        # gradient_clip_val=1.0
        # num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    main()
