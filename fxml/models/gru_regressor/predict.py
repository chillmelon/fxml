import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from fxml.models.gru_regressor.model import GRURegressorModule


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(config: DictConfig):
    df = pd.read_pickle(config["data"]["dataset_path"])

    models_dir = Path(
        f"lightning_logs/{config['model']['name']}_{Path(config["data"]["dataset_path"]).stem}"
    )
    version_list = sorted(os.listdir(models_dir))

    print(version_list[-1])
    checkpoint_path = (
        models_dir / version_list[-1] / "checkpoints" / "best_checkpoint.ckpt"
    )

    # df = df.iloc[-1000:].copy()
    close = np.log(df["close"].to_numpy())
    len_df = len(close)
    lookback = config["data"]["sequence_length"]

    list_slices = [close[i : len_df - lookback + i] for i in range(0, lookback)]
    X = torch.Tensor(
        np.array(np.vstack(list_slices).T.reshape((-1, lookback, 1)), dtype=np.float32)
    )

    y_true = np.array(close[lookback:], dtype=np.float32)

    model = GRURegressorModule.load_from_checkpoint(checkpoint_path)
    model.to("cpu")
    model.eval()
    y_pred = model(X)[1].detach().numpy().squeeze()
    df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})

    plt.figure(figsize=(6, 6))
    plt.scatter(df["Actual"], df["Predicted"], alpha=0.5)
    plt.plot(
        [df["Actual"].min(), df["Actual"].max()],
        [df["Actual"].min(), df["Actual"].max()],
        color="red",
        linestyle="--",
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.show()


if __name__ == "__main__":
    main()
