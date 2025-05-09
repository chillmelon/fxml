import torch
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from lightning.pytorch import Trainer
from datamodules.data_module import ForexDataModule
from dataset.dataset import ForexDataset
from models.gru_model import GRUModule
from utils import get_sequence_start_indices

# === CONFIG ===
CHECKPOINT_PATH = r'lightning_logs\prob_gru\version_6\checkpoints\best_checkpoint.ckpt'
SCALER_PATH='standard_scaler.pkl'
DATA_PATH = "./data/processed/usdjpy-bar-2020-01-01-2024-12-31_processed.pkl"
FEATURES_COLS = ['close_return']
SEQUENCE_LENGTH = 30
TARGET_COLS = ['prob_down', 'prob_flat', 'prob_up']
HORIZON = 1
STRIDE = 1

def plot_prediction(y_pred, y_true=None, title="Prediction"):
    # mu = mu.squeeze().cpu()


    plt.figure(figsize=(12, 6))
    plt.plot(y_pred, label="Predicted", color="blue")
    if y_true is not None:
        plt.plot(y_true.cpu(), label="Ground Truth", color="black", linestyle="--")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_pickle(DATA_PATH)

    IDs = get_sequence_start_indices(
        df,
        sequence_length=SEQUENCE_LENGTH,
        horizon=HORIZON,
        stride=STRIDE,
        group_col='time_group',
    )
    # Initialize Data Module
    dm = ForexDataModule(
        data=df,
        IDs=IDs,
        sequence_length=SEQUENCE_LENGTH,
        target=TARGET_COLS,
        features=FEATURES_COLS,
        horizon=HORIZON,
        batch_size=64,
        val_split=0.2,
        num_workers=0,
    )

    dm.setup(stage="test")

    model = GRUModule.load_from_checkpoint(CHECKPOINT_PATH)

    # Run test
    trainer = Trainer(logger=False, enable_checkpointing=False)
    trainer.test(model, datamodule=dm)

    # Predict on one batch
    val_loader = dm.test_dataloader()
    x, y, _ = next(iter(val_loader))
    with torch.no_grad():
        _, probs = model(x)
        pred_classes = torch.argmax(probs, dim=1)

        plot_prediction(pred_classes, y_true=torch.argmax(y, dim=1))

if __name__ == "__main__":
    main()
