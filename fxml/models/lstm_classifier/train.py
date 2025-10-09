import pandas as pd
import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from fxml.data.datamodules.event_based_datamodule import EventBasedDataModule
from fxml.data.datasets.direction_dataset import DirectionDataset
from fxml.models.lstm_classifier.model import LSTMClassifierModule
from fxml.utils import get_device


def main():
    config = yaml.safe_load(open("configs/lstm_classifier.yaml", "r"))
    df = pd.read_pickle(config["data"]["dataset_path"])
    label = pd.read_pickle(config["data"]["label_path"])

    dm = EventBasedDataModule(
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

    model = LSTMClassifierModule(
        n_features=len(config["data"]["features"]),
        output_size=3,
        n_hidden=config["model"]["n_hidden"],
        n_layers=config["model"]["n_layers"],
        dropout=config["model"]["dropout"],
    )

    logger = TensorBoardLogger("lightning_logs", name=f"{config['model']['name']}")

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
        accelerator=get_device(),
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
