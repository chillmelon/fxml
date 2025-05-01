from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
import torch
import yaml
import pandas as pd
from matplotlib import pyplot as plt

from datamodules.fx_clas_dm import ForexClassificationDataModule
from models.gru_classifier import ForexGRUClassifier
import pandas as pd

from models.lstm_classifier import LSTMClassifierModule

# DATA_PATH = r'data\processed\usdjpy-20200101-20241231.csv'
DATA_PATH = './data/processed/usd-jpy-2024.csv'
PKL_PATH = './data/processed/usd-jpy-2024.pkl'
HORIZON=1

def main():
    # Initialize Data Module
    dm = ForexClassificationDataModule(
        data_path=DATA_PATH,
        sequence_length=30,
        target='label',
        features=['close_delta'],
        target_horizon=HORIZON,
        batch_size=64,
        val_split=0.2,
    )

    # Initialize GRU module
    model = LSTMClassifierModule(
        n_features=1,
        n_classes=3,
        n_hidden=64,
        n_layers=2,
        dropout=0.2
    )
    # Start Logger
    logger = TensorBoardLogger("lightning_logs", name="forex_lstm")

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
    # Training
    trainer = Trainer(
        profiler=profiler,
        callbacks=[checkpoint_callback, early_stopping],
        max_epochs=200,
        logger=logger,
    )
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    main()
