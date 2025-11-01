"""Base regressor module for all regression models.

This module provides common training/validation/test logic to eliminate code duplication
across different regressor architectures (Baseline, LSTM, GRU, Transformer, T2V).
"""

import lightning as pl
import matplotlib.pyplot as plt
import torch
from torch import nn


class BaseRegressorModule(pl.LightningModule):
    """Base Lightning module for regression tasks.

    This class provides:
    - Common training/validation/test steps with MSE loss
    - Validation plotting (predicted vs actual, residuals)
    - Configurable optimizer and scheduler

    Subclasses must implement:
    - build_model(): Returns the nn.Module for the specific architecture
    """

    def __init__(
        self,
        lr=1e-3,
        optimizer_type="adam",  # "Adam" or "AdamW"
        weight_decay=1e-4,
        scheduler_step_size=10,
        scheduler_gamma=0.5,
        enable_plotting=True,
    ):
        """Initialize base regressor module.

        Args:
            lr: Learning rate for optimizer
            optimizer_type: Type of optimizer ("Adam" or "AdamW")
            weight_decay: Weight decay for regularization
            scheduler_step_size: Step size for learning rate scheduler
            scheduler_gamma: Multiplicative factor for learning rate decay
            enable_plotting: Whether to enable validation plotting
        """
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.enable_plotting = enable_plotting

        # Loss criterion
        self.criterion = nn.MSELoss()

        # For validation plotting
        self.val_preds = []
        self.val_labels = []

        # Subclass must set self.model in their __init__
        self.model = None

    def on_train_start(self):
        """Log hyperparameters to TensorBoard at the start of training."""
        if self.logger:
            # Get all hyperparameters saved by save_hyperparameters()
            hparams = dict(self.hparams)

            # Define metric for hyperparameter comparison
            metric_dict = {"test_loss": -1}

            # Log to TensorBoard
            self.logger.log_hyperparams(hparams, metric_dict)

    def build_model(self):
        """Build and return the model architecture.

        This method must be implemented by subclasses.

        Returns:
            nn.Module: The model architecture
        """
        raise NotImplementedError("Subclasses must implement build_model()")

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features)

        Returns:
            Predictions of shape (batch_size, output_size)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Subclass must set self.model")
        return self.model(x)

    def _common_step(self, batch, stage):
        """Common step for training/validation/test.

        Args:
            batch: Tuple of (x, y, *) where x is input and y is target
            stage: One of "train", "val", "test"

        Returns:
            Tuple of (predictions, labels, loss)
        """
        assert stage in ("train", "val", "test")

        x, y = batch

        # Forward pass
        preds = self(x)

        # Compute loss
        loss = self.criterion(preds, y)

        # Log loss
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return preds, y, loss

    def training_step(self, batch, batch_idx):
        preds, y, loss = self._common_step(batch, "train")
        return {"loss": loss, "outputs": preds, "labels": y}

    def validation_step(self, batch, batch_idx):
        preds, y, loss = self._common_step(batch, "val")

        # Store predictions and labels for plotting
        if self.enable_plotting:
            self.val_preds.append(preds)
            self.val_labels.append(y)

        return {"loss": loss, "outputs": preds, "labels": y}

    def on_validation_epoch_end(self):
        if not self.enable_plotting or not self.val_preds:
            return

        # Concatenate all predictions and labels
        preds = torch.cat(self.val_preds).detach().cpu().flatten()
        labels = torch.cat(self.val_labels).detach().cpu().flatten()

        # Clear for next epoch
        self.val_preds.clear()
        self.val_labels.clear()

        # Create figure with two subplots
        fig, ax = plt.subplots(1, 2, figsize=(16, 5))

        # Plot 1: Predicted vs Actual
        ax[0].scatter(labels, preds, alpha=0.5)
        ax[0].plot(
            [labels.min(), labels.max()], [labels.min(), labels.max()], "r--", lw=2
        )
        ax[0].set_xlabel("True values")
        ax[0].set_ylabel("Predicted values")
        ax[0].set_title("Predicted vs True")

        # Plot 2: Residuals
        residuals = preds - labels
        ax[1].hist(residuals, bins=40, alpha=0.7, color="steelblue")
        ax[1].axvline(0, color="r", linestyle="--")
        ax[1].set_title("Residual Distribution")
        ax[1].set_xlabel("Residual (Pred - True)")
        ax[1].set_ylabel("Count")

        plt.tight_layout()

        # Log figure to TensorBoard
        self.logger.experiment.add_figure(
            "Validation/Regression_Evaluation", fig, self.current_epoch
        )
        plt.close(fig)

    def test_step(self, batch, batch_idx):
        preds, y, loss = self._common_step(batch, "test")
        return {"loss": loss, "outputs": preds, "labels": y}

    def on_test_end(self):
        """Log test loss as hp_metric after testing completes."""
        if self.logger and hasattr(self.trainer.callback_metrics, "__getitem__"):
            test_loss = self.trainer.callback_metrics.get("test_loss", None)
            if test_loss is not None:
                self.logger.log_metrics(
                    {"test_loss": test_loss.item()}, step=self.current_epoch
                )

    def configure_optimizers(self):
        # Select optimizer type
        if self.optimizer_type.lower() == "adamw":
            opt = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == "adam":
            opt = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        # Configure scheduler
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma
        )

        return [opt]
        # [sched]
