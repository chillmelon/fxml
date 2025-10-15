import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from fxml.models.model import build_model


def extract_sequences(data: pd.DataFrame, t_events, lookback: int):
    """Extract valid sequences and events for batch processing."""
    sequences, valid_events = [], []
    data_values, data_index = data.values.astype(np.float32), data.index

    for t in t_events:
        try:
            if t in data_index:
                end_loc = data_index.get_loc(t)
                start_loc = end_loc - lookback
                if start_loc >= 0:
                    seq = data_values[start_loc:end_loc]
                    if len(seq) == lookback:
                        sequences.append(seq)
                        valid_events.append(t)
        except:
            continue

    return (np.stack(sequences), valid_events) if sequences else (None, [])


def get_side_from_model_batch(
    model,
    data: pd.DataFrame,
    labels: pd.DataFrame,
    lookback: int = 24,
    device: str = "cpu",
    batch_size: int = 64,
):
    """Batch process model predictions for time series events."""
    model.eval().to(device)

    sequences, valid_events = extract_sequences(data, labels.index, lookback)

    if sequences is None:
        return pd.DataFrame()

    predictions, probabilities = [], []
    n_batches = (len(sequences) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(sequences), batch_size), total=n_batches, desc="Processing batches"
    ):
        batch = sequences[i : i + batch_size]
        x_tensor = torch.tensor(batch, device=device)
        with torch.no_grad():
            logits = model(x_tensor)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    result_df = pd.DataFrame(probabilities, index=valid_events)
    result_df.columns = [f"prob_{i}" for i in range(len(result_df.columns))]
    result_df["prediction"] = predictions
    result_df["pred_side"] = result_df["prediction"] - 1

    return result_df


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config: DictConfig):

    models_dir = Path(
        f"./lightning_logs/{config['model']['name']}_{Path(config["data"]["label_path"]).stem}"
    )

    version_list = sorted(os.listdir(models_dir))

    print(version_list)
    checkpoint_path = (
        models_dir / version_list[-1] / "checkpoints" / "best_checkpoint.ckpt"
    )

    df = pd.read_pickle(config["data"]["dataset_path"])
    label = pd.read_pickle(config["data"]["label_path"])
    features = config["data"]["features"]
    features_df = df[features]

    model = build_model(config["model"]["name"], config).__class__
    model = model.load_from_checkpoint(checkpoint_path)

    model.to("cpu")
    model.eval()

    predictions = get_side_from_model_batch(
        model=model,
        data=features_df,
        labels=label,
        lookback=config["data"]["sequence_length"],
        device="mps",
        batch_size=1024,
    )

    label = label.join(predictions)
    label.dropna(inplace=True)
    label["confidence"] = label.apply(
        lambda x: x[f"prob_{int(x['prediction'])}"], axis=1
    )

    predictions_path = (
        Path("./data/predictions")
        / f"{config['model']['name']}_{Path(config["data"]["label_path"]).stem}.pkl"
    )

    label.to_pickle(predictions_path)


if __name__ == "__main__":
    main()
