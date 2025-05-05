import torch
import joblib
import matplotlib.pyplot as plt
from lightning.pytorch import Trainer
from models.gru_prob_regr import ProbabilisticGRURegressor
from datamodules.fx_regr_dm import ForexRegressionDataModule

# === CONFIG ===
CHECKPOINT_PATH = r'lightning_logs\prob_gru_multi\version_4\checkpoints\best_checkpoint.ckpt'
DATA_PATH = "./data/processed/usdjpy-bar-2024-01-01-2024-12-31_processed.pkl"
FEATURES_COLS = ['close_return', 'volume']
SEQUENCE_LENGTH = 30
TARGET_COL = 'close_return'
TARGET_HORIZON = 3

def plot_prediction_with_uncertainty(mu, sigma, y_true=None, title="Prediction with Confidence"):
    mu = mu.squeeze().cpu()
    sigma = sigma.squeeze().cpu()
    upper = mu + 1.96 * sigma
    lower = mu - 1.96 * sigma

    plt.figure(figsize=(12, 6))
    plt.plot(mu, label="Predicted μ (mean)", color="blue")
    plt.fill_between(range(len(mu)), lower, upper, alpha=0.3, label="95% Confidence Interval")
    if y_true is not None:
        plt.plot(y_true.squeeze().cpu(), label="Ground Truth", color="black", linestyle="--")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Load model from checkpoint
    model = ProbabilisticGRURegressor.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()
    model.freeze()

    # Setup datamodule
    dm = ForexRegressionDataModule(
        data_path=DATA_PATH,
        sequence_length=SEQUENCE_LENGTH,
        target=TARGET_COL,
        features=FEATURES_COLS,
        target_horizon=TARGET_HORIZON,
        batch_size=64,
        val_split=0.2,
        num_workers=0
    )
    dm.prepare_data()
    dm.setup(stage="test")

    # Run test
    trainer = Trainer(logger=False, enable_checkpointing=False)
    trainer.test(model, datamodule=dm)

    # Predict on one batch
    val_loader = dm.val_dataloader()
    x, y = next(iter(val_loader))
    with torch.no_grad():
        mu, log_sigma = model(x)
        sigma = torch.exp(log_sigma)
        plot_prediction_with_uncertainty(mu, sigma, y_true=y)

if __name__ == "__main__":
    main()
