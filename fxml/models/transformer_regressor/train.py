import pandas as pd
import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from fxml.data.datamodules.return_datamodule import ReturnDataModule
from fxml.models.lstm_regressor.lstm_regressor import LSTMRegressorModule
from fxml.models.transformer_regressor.model import TransformerRegressorModule


def main():
    config = yaml.safe_load(open("configs/transformer_regressor.yaml", "r"))
    df = pd.read_pickle(config["data"]["dataset_path"])
    label = pd.read_pickle(config["data"]["label_path"])

    dm = ReturnDataModule(
        data=df,
        labels=label,
        sequence_length=config["data"]["sequence_length"],
        features=config["data"]["features"],
        target=config["data"]["target"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        val_split=config["training"]["val_split"],
        shuffle=config["training"]["shuffle"],
    )

    model = TransformerRegressorModule(
        n_features=len(config["data"]["features"]),
        output_size=1,
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        n_layers=config["model"]["n_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        pool=config["model"]["pool"],
    )

    logger = TensorBoardLogger(
        "lightning_logs", name=f"models/{config['model']['name']}"
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
    )
    torch.set_float32_matmul_precision("high")
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
