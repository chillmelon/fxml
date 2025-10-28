"""Base classifier module for all classification models."""

import lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torchmetrics.classification import ConfusionMatrix, MulticlassAccuracy


class BaseClassifierModule(pl.LightningModule):
    """
    Base Lightning Module for all classification models.

    This class implements common functionality for classification tasks:
    - Training, validation, and test steps
    - Loss calculation with optional label smoothing
    - Accuracy metrics tracking
    - Confusion matrix visualization
    - Configurable optimizer (Adam or AdamW) with StepLR scheduler

    Subclasses should:
    1. Call super().__init__() with common parameters
    2. Create self.model with their specific architecture
    3. Optionally override methods for custom behavior

    Args:
        n_features: Number of input features
        output_size: Number of output classes
        lr: Learning rate (default: 1e-3)
        label_smoothing: Label smoothing factor for cross-entropy loss (default: 0.0)
        optimizer: Optimizer type, "adam" or "adamw" (default: "adamw")
        weight_decay: Weight decay for AdamW optimizer (default: 1e-4)
    """

    def __init__(
        self,
        n_features: int,
        output_size: int,
        lr: float = 1e-3,
        label_smoothing: float = 0.0,
        optimizer: str = "adamw",
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.output_size = output_size
        self.lr = lr

        # Model to be set by subclass
        self.model = None

        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Metrics
        self.val_acc = MulticlassAccuracy(num_classes=output_size)
        self.test_acc = MulticlassAccuracy(num_classes=output_size)

        # Storage for confusion matrix computation
        self.val_preds = []
        self.val_labels = []

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)  # logits

    def training_step(self, batch, batch_idx):
        """Training step: compute loss and log metrics."""
        x, y = batch
        y = y.squeeze().long()
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step: compute loss, accuracy, and store predictions."""
        x, y = batch
        y = y.squeeze().long()
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        # Store predictions and labels for confusion matrix
        self.val_preds.append(preds)
        self.val_labels.append(y)

        return loss

    def on_validation_epoch_end(self):
        """Compute and plot confusion matrix at the end of validation epoch."""
        if len(self.val_preds) == 0:
            return

        val_preds = torch.cat(self.val_preds)
        val_labels = torch.cat(self.val_labels)

        conf_mat = ConfusionMatrix(task="multiclass", num_classes=self.output_size)
        cm = conf_mat(val_preds.cpu(), val_labels.cpu())

        self.plot_confusion_matrix(cm)

        # Clear stored predictions and labels for the next epoch
        self.val_preds.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        """Test step: compute loss and accuracy."""
        x, y = batch
        y = y.squeeze().long()
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def plot_confusion_matrix(self, cm):
        """Plot and log confusion matrix to TensorBoard."""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")

        # Log confusion matrix to TensorBoard
        self.logger.experiment.add_figure("Confusion Matrix", fig, self.current_epoch)
        plt.close(fig)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Select optimizer based on hyperparameter
        if self.hparams.optimizer == "adamw":
            opt = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:  # adam
            opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        # StepLR scheduler: reduce LR by 0.5 every 10 epochs
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

        return [opt], [sched]
