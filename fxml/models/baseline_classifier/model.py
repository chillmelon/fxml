import lightning as pl
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchmetrics.classification import ConfusionMatrix, MulticlassAccuracy


class BaselineClassifier(nn.Module):
    """Simple baseline classifier using mean pooling and feedforward layers.

    This model serves as a baseline by:
    1. Mean pooling across the sequence dimension
    2. Passing through a simple feedforward network

    This provides a non-sequential baseline to compare against LSTM/Transformer models.
    """

    def __init__(
        self,
        n_features,
        output_size,
        n_hidden=64,
        dropout=0.1,
    ):
        super().__init__()

        # Simple feedforward network after mean pooling
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(n_hidden, n_hidden // 2)
        self.fc_out = nn.Linear(n_hidden // 2, output_size)

    def forward(self, x):
        # x: (B, T, n_features)
        # Mean pooling across time dimension
        x = x.mean(dim=1)  # (B, n_features)

        # Feedforward network
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc_out(x)  # (B, output_size)

        return logits


class BaselineClassifierModule(pl.LightningModule):
    def __init__(
        self,
        n_features=1,
        output_size=3,
        n_hidden=64,
        dropout=0.1,
        label_smoothing=0.0,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.output_size = output_size

        self.model = BaselineClassifier(
            n_features=n_features,
            output_size=output_size,
            n_hidden=n_hidden,
            dropout=dropout,
        )

        self.val_preds = []
        self.val_labels = []

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.val_acc = MulticlassAccuracy(num_classes=output_size)
        self.test_acc = MulticlassAccuracy(num_classes=output_size)
        self.lr = lr

    def forward(self, x):
        return self.model(x)  # logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze().long()
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
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
        val_preds = torch.cat(self.val_preds)
        val_labels = torch.cat(self.val_labels)

        conf_mat = ConfusionMatrix(task="multiclass", num_classes=self.output_size)
        # Compute confusion matrix
        cm = conf_mat(val_preds.cpu(), val_labels.cpu())

        # Plot confusion matrix
        self.plot_confusion_matrix(cm)

        # Clear stored predictions and labels for the next epoch
        self.val_preds.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze().long()
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")

        # Log confusion matrix to TensorBoard
        self.logger.experiment.add_figure("Confusion Matrix", fig, self.current_epoch)
        plt.close(fig)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        return [opt], [sched]
