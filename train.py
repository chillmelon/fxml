from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
import torch
import yaml
import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd

from datamodules.fx_clas_dm import ForexClassificationDataModule
from datamodules.fx_regr_dm import ForexRegressionDataModule
from dataset.forex_regr_dataset import ForexRegressionDataset
from models.gru_regr import GRURegressorModule
from models.lstm_classifier import LSTMClassifierModule
from models.gru_classifier import GRUClassifierModule
from models.transformer_classifier import TransformerClassifierModule

# DATA_PATH = r'data\processed\usdjpy-20200101-20241231.csv'
DATA_PATH = './data/processed/usd-jpy-2024.csv'
PKL_PATH = './data/processed/usdjpy-bar-2024-01-01-2024-12-31_processed.pkl'
HORIZON=1
FEATURES_COLS = ['close_return']
def main():
    # Initialize Data Module
    dm = ForexRegressionDataModule(
        data_path=PKL_PATH,
        sequence_length=5,
        target='close_return',
        features=FEATURES_COLS,
        target_horizon=HORIZON,
        batch_size=64,
        # split_method='stratified',
        val_split=0.2,
        num_workers=0,
    )

    # Initialize GRU module
    model = GRURegressorModule(
        n_features=len(FEATURES_COLS),
        horizon=1,
        n_hidden=64,
        n_layers=2,
        dropout=0.8,
        lr=1e-4
    )
    # Start Logger
    logger = TensorBoardLogger("lightning_logs", name="forex_lstm_regr")

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
        # num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    main()
