from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler
import torch
import yaml
import pandas as pd
from matplotlib import pyplot as plt

from dataset.forex_dataset import ForexDataset
from models.fx_datamodule import ForexDataModule
from models.gru_module import ForexGRU
import pandas as pd

DATA_PATH = './data/processed/usdjpy-20200101-20241231.csv'
PKL_PATH = './data/processed/usd-jpy-2024.pkl'
HORIZON=1

def main():
    # Initialize Data Module
    dm = ForexDataModule(
        data_path=PKL_PATH,
        sequence_length=30,
        target='close_pct_delta',
        features=['close_pct_delta'],
        target_horizon=HORIZON,
        batch_size=64,
        val_split=0.2,
    )

    # Initialize GRU module
    model = ForexGRU(horizon=HORIZON)
    # Start Logger
    logger = TensorBoardLogger("lightning_logs", name="forex_gru")

    profiler = AdvancedProfiler(filename='profiler.txt')
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=3,
        verbose=True
    )

    # Training
    trainer = Trainer(
        profiler=profiler,
        callbacks=[early_stopping],
        max_epochs=200,
        logger=logger
    )
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    main()
