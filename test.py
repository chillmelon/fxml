import torch
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from dataset.dataset import ForexDataset
from models.gru_model import GRUModule
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils import get_sequence_start_indices

# === CONFIG ===
CHECKPOINT_PATH = r'lightning_logs\prob_gru\version_9\checkpoints\best_checkpoint.ckpt'
SCALER_PATH='standard_scaler.pkl'
DATA_PATH = "./data/processed/usdjpy-bar-2025-01-01-2025-05-12_processed.pkl"
FEATURES_COLS = ['close_return']
SEQUENCE_LENGTH = 30
TARGET_COLS = ['label']
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

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_class_predictions(y_pred, y_true):
    plt.figure(figsize=(12, 4))
    plt.scatter(range(len(y_pred)), y_pred, label="Predicted", marker='o', alpha=0.6)
    plt.scatter(range(len(y_true)), y_true, label="Ground Truth", marker='x', alpha=0.6)
    plt.title("Classification Predictions vs Ground Truth")
    plt.xlabel("Sample Index")
    plt.ylabel("Class")
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

    test_dataset = ForexDataset(
        df, IDs, SEQUENCE_LENGTH, HORIZON, FEATURES_COLS, TARGET_COLS
    )


    model = GRUModule.load_from_checkpoint(CHECKPOINT_PATH)


    # Predict on one batch
    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False
    )

    model.to('cpu')
    x, y, _ = next(iter(test_loader))
    with torch.no_grad():
        _, probs = model(x)
        pred_classes = torch.argmax(probs, dim=1)

        plot_confusion_matrix(pred_classes, y)

if __name__ == "__main__":
    main()
