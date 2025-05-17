from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
import torch
import yaml
import pandas as pd
from matplotlib import pyplot as plt

from datamodules.data_module import ForexDataModule
from models.gru_model import GRUModule
from models.transformer_model import TransformerModule
from utils import get_sequence_start_indices

PKL_PATH = r'data\processed\usdjpy-bar-2020-01-01-2024-12-31_processed.pkl'
SEQUENCE_LENGTH=30
HORIZON=1
STRIDE=5
FEATURES_COLS = ['close_log_return', 'ret_mean_5']
TARGET_COLS = ['label']
def main():
    df = pd.read_pickle(PKL_PATH)


    IDs = get_sequence_start_indices(
        df,
        sequence_length=SEQUENCE_LENGTH,
        horizon=HORIZON,
        stride=STRIDE,
        group_col='time_group',
    )
    # Initialize Data Module
    dm = ForexDataModule(
        data=df,
        IDs=IDs,
        sequence_length=SEQUENCE_LENGTH,
        target=TARGET_COLS,
        features=FEATURES_COLS,
        horizon=HORIZON,
        batch_size=2048,
        val_split=0.2,
        num_workers=0,
    )

    # Initialize GRU module
    # model = GRUModule(
    #     n_features=len(FEATURES_COLS),
    #     output_size=3,
    #     n_hidden=64,
    #     n_layers=2,
    #     dropout=0.3,
    # )
    model = TransformerModule(
        n_features=len(FEATURES_COLS),
        output_size=3,

    )

    # Start Logger
    logger = TensorBoardLogger("lightning_logs", name="transformer")

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
    # torch.set_float32_matmul_precision('high')
    # Training
    trainer = Trainer(
        # accelerator="gpu",
        # precision='16-mixed',
        profiler=profiler,
        callbacks=[checkpoint_callback, early_stopping],
        max_epochs=200,
        logger=logger,
        gradient_clip_val=1.0
        # num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    main()
