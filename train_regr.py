import os
from pathlib import Path

import hydra
import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from omegaconf import DictConfig, OmegaConf

from fxml.data.datamodules.multistep_regr_datamodule import MultiStepRegrDataModule
from fxml.models.model import build_model
from fxml.utils import get_device


@hydra.main(version_base=None, config_path="./configs", config_name="tune_t2v_xfmr")
def main(cfg: DictConfig):
    train_data = pd.read_pickle(cfg.data.train_path)
    test_data = pd.read_pickle(cfg.data.test_path)

    dm = MultiStepRegrDataModule(
        train_data,
        test_data,
        feature_cols=cfg.data.time_features + cfg.data.features,
        target_col=cfg.data.target,
        lookback=cfg.data.lookback,
        lookforward=cfg.data.lookforward,
        val_split=cfg.training.val_split,
        batch_size=cfg.training.batch_size,
    )

    model = build_model(cfg.model.name, cfg)
    logger = TensorBoardLogger(
        "lightning_logs",
        name=f"{cfg.model.name}_{Path(cfg.data.train_path).stem}",
        default_hp_metric=False,
    )

    profiler = SimpleProfiler(filename="profiler")

    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=5, verbose=True
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
