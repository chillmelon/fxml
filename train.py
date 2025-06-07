from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
import torch
import yaml
import pandas as pd
from matplotlib import pyplot as plt

from datamodules.data_module import ForexDataModule
from dataset.dataset import ForexDataset
from models.gru_model import GRUModule
from models.transformer_model import TransformerModule
from models.t2v_transformer_model import T2VTransformerModule
from utils import get_sequence_start_indices

PKL_PATH = r'data\normalized\dukascopy-usdjpy-15m-2020-01-01-2024-12-31_normalized.pkl'
SEQUENCE_LENGTH=24
HORIZON=1
STRIDE=1
TIME_COLS = [
    # 'timestamp',
    # 'hour',
    # 'dow',
    # 'dom',
    # 'month',
    # 'open',
    # 'high',
    # 'low',
    # 'close',
]
FEATURES_COLS = [
    # Basic Data
    'close_log_return',
    'log_volume',
    # 'spread',

    # Time feature
    # 'hour_cos',
    # 'dow_cos',
    # 'dom_cos',

    # # Other
    # 'ret_mean_5',
    # 'ret_mean_10',

    # # TA
    # 'rsi_14',
    # 'ema_21',
    # 'sma_50',
    # 'atr_14',

    # 'bb_upper',
    # 'bb_lower',
    # 'bb_mavg',
    # 'bb_width',

    # 'donchian_upper',
    # 'donchian_lower',
    # 'donchian_mid',

    'stoch_k',
    'stoch_d',

    'macd',
    'macd_signal',
    'macd_diff',
]

TARGET_COLS = ['3b_train_label']


def main():
    df = pd.read_pickle(PKL_PATH)
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df = df[df['datetime'].dt.year >= 2021]
    print(df.head())
    print(f"Data shape after filtering: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    df['timestamp'] = df['timestamp'].astype('int64') / 1e18
    # Check for NaN values
    nan_counts = df.isna().sum()
    print(f"NaN counts in features and targets:\n{nan_counts}")
    print(df['train_label'].value_counts(normalize=True))
    df = df[TIME_COLS+ FEATURES_COLS + TARGET_COLS]

    dm = ForexDataModule(
        data=df,
        features=TIME_COLS+FEATURES_COLS,
        target=TARGET_COLS,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        batch_size=1024,
        val_split=0.2,
        # num_workers=19,
    )

    dm.setup()

    # Initialize GRU module
    # model = GRUModule(
    #     n_features=len(FEATURES_COLS),
    #     output_size=3,
    #     n_hidden=256,
    #     n_layers=3,
    #     dropout=0.7,
    # )

    # model = TransformerModule(
    #     n_features=len(FEATURES_COLS),
    #     output_size=3,
    #     num_layers=3,
    #     d_model=64,
    #     nhead=4,
    #     dim_feedforward=256,
    #     dropout=0.3
    # )

    model = T2VTransformerModule(
        n_time=len(TIME_COLS),
        n_features=len(FEATURES_COLS),
        output_size=3,
        num_layers=2,
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        kernel_size=1,
        dropout=0.7,
        # label_smoothing=0.05
    )

    # Start Logger
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
