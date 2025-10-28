from pathlib import Path

import hydra
import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from omegaconf import DictConfig

from fxml.data.datamodules.clf_datamodule import ClassificationDataModule
from fxml.models.model import build_model
from fxml.utils import get_device


@hydra.main(version_base=None, config_path="./configs", config_name="ts_tbm_clf")
def main(config: DictConfig):
    train_data = pd.read_pickle(config["data"]["train_path"])
    test_data = pd.read_pickle(config["data"]["test_path"])

    dm = ClassificationDataModule(
        train_data,
        test_data,
        feature_cols=config["data"]["time_features"] + config["data"]["features"],
        target_col="class",
        lookback=config["data"]["lookback"],
        stride=config["data"]["stride"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        val_split=0.5,
    )

    model = build_model(config["model"]["name"], config)
    logger = TensorBoardLogger(
        "lightning_logs",
        name=f"{config['model']['name']}_{Path(config["data"]["train_path"]).stem}",
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
        accelerator=get_device(),
        devices=1,
        profiler=profiler,
        callbacks=[checkpoint_callback, early_stopping],
        max_epochs=config["training"]["num_epochs"],
        logger=logger,
    )
    torch.set_float32_matmul_precision("high")
    trainer.fit(model, datamodule=dm)

    last_model_path = checkpoint_callback.last_model_path
    best_model_path = checkpoint_callback.best_model_path
    _use_model_path = last_model_path if best_model_path == "" else best_model_path
    print("use checkpoint:", _use_model_path)

    # run_test
    trainer.test(
        model=model if _use_model_path == "" else None,
        datamodule=dm,
        ckpt_path=_use_model_path,
    )


if __name__ == "__main__":
    main()
