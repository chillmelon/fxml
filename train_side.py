import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from datamodules.event_based_data_module import EventBasedDataModule
from models.classification.simple_transformer_model import SimpleTransformerModule
from models.classification.t2v_transformer_model import T2VTransformerModule
from utils import build_file_paths_from_config, get_device, load_config

# Default configurable parts (can be overridden by config)
MODEL_NAME = "simple_transformer"
SEQUENCE_LENGTH = 120
# Feature configuration
TIME_COLS = [
    "hour",
    "dow",
    "dom",
    "month",
]

FEATURES_COLS = [
    "hour_cos",
    "dow_cos",
    "dom_cos",
    "month_cos",
    "close_log_return",
    "ret_mean_5",
    "ret_mean_15",
    "rv5",
    "sqrt_rv5",
    "rv15",
    "sqrt_rv15",
    "rv50",
    "sqrt_rv50",
    "ema5_slope",
    "ema20_slope",
    "ema50_slope",
    "ema100_slope",
    "close_above_ema20",
    "log_atr14",
    "log_atr60",
    "atr14_adjusted_return",
    "adx14",
    "plus_di14",
    "minus_di14",
    "rsi14",
    "rsi14_slope",
    "macd_diff",
    "bb_width",
    "bb_position",
    "dc20_width",
    # 'close_above_dc20_mid',
    # 'dc20_breakout',
    # 'dc20_breakdown',
]

TARGET_COL = "bin_class"


def main(config_path="config/config.yaml"):
    # Load config and build paths
    config = load_config(config_path)
    paths, sample_event, label_event = build_file_paths_from_config(config)

    print(f"Loading data from:")
    print(f"  Normalized: {paths['normalized']}")
    print(f"  Labels: {paths['direction_labels']}")

    # Load data
    df = pd.read_pickle(paths["normalized"])
    label_df = pd.read_pickle(paths["direction_labels"])

    print(df.loc[label_df.index, FEATURES_COLS].head())

    print(label_df[TARGET_COL].value_counts())
    X_COLS = TIME_COLS + FEATURES_COLS

    dm = EventBasedDataModule(
        data=df,
        labels=label_df,
        sequence_length=SEQUENCE_LENGTH,
        # features=X_COLS,
        features=FEATURES_COLS,
        target=TARGET_COL,
        batch_size=256,
        balanced_sampling=False,
    )

    model = SimpleTransformerModule(
        n_features=len(FEATURES_COLS),
        output_size=3,
        num_layers=2,
        d_model=64,
        nhead=4,
        dim_feedforward=256,
        dropout=0.4,
        label_smoothing=0.0,
        pool="last",
        use_class_weights=True,
    )

    logger = TensorBoardLogger(
        "lightning_logs", name=f"{MODEL_NAME}-{sample_event}-{label_event}"
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

    device = get_device()

    # Training
    trainer = Trainer(
        accelerator=device,
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
