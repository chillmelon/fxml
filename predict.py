from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from fxml.data.datamodules.multistep_regr_datamodule import create_multistep_sequences
from fxml.data.normalize_data import normalize
from fxml.models.model import build_model


@hydra.main(version_base=None, config_path="./configs", config_name="ts_seq2seq")
def main(config: DictConfig):
    scaler_cfg = config.scaler
    scaler_dir = Path("data/processed/scalers")

    # Load Data
    df = pd.read_pickle(config.data.test_path)
    # df = normalize(df, config.train_data, scaler_cfg, str(scaler_dir))
    features = config.data.features
    targets = config.data.target
    lookback = config.data.lookback
    lookforward = config.data.lookforward

    windows, _ = create_multistep_sequences(
        df, lookback, lookforward, features, [targets[0]]
    )

    # Load Model
    checkpoint_path = config.best_checkpoint_path
    model = build_model(config.model.name, config).__class__
    model = model.load_from_checkpoint(checkpoint_path)
    model.to("cpu")
    model.eval()

    # Load Scaler
    scaler_type = scaler_cfg[targets[0]].type

    scaler_path = scaler_dir / config.train_data / f"{targets[0]}_{scaler_type}.pkl"
    scaler = joblib.load(scaler_path)

    # Inference
    results = []
    for i in range(0, len(windows), 1024):
        sequence = windows[i : i + 1024]
        X = torch.tensor(sequence, dtype=torch.float32)
        y = model(X).detach().numpy()[:, :, None]
        # y = np.array(list(map(scaler.inverse_transform, y)))
        results.append(y[:, :, 0])

    results = np.vstack(results)
    results = scaler.inverse_transform(results)

    horizons = pd.DataFrame(
        results,
        columns=pd.Index([f"h{n}" for n in range(lookforward)]),
        index=df.index[lookback - 1 : -lookforward],
    )

    horizons.to_pickle(
        Path(f"data/predictions") / f"test_{config.model.name}_h{lookforward}.pkl"
    )


if __name__ == "__main__":
    main()
