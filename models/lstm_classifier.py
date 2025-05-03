import torch
from torch import nn
import lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

class LSTMModel(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden, n_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        # self.batch_norm = nn.BatchNorm1d(n_hidden)
        self.classifier = nn.Linear(n_hidden, n_classes)


    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        # normalized = self.batch_norm(last_hidden)
        return self.classifier(last_hidden)


class LSTMClassifierModule(pl.LightningModule):
    def __init__(self, n_features=1, n_classes=3, n_hidden=64, n_layers=2, dropout=0.0, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = LSTMModel(
            n_features=self.hparams.n_features,
            n_classes=self.hparams.n_classes,
            n_hidden=self.hparams.n_hidden,
            n_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = MulticlassAccuracy(num_classes=n_classes)
        self.val_acc = MulticlassAccuracy(num_classes=n_classes)
        self.test_acc = MulticlassAccuracy(num_classes=n_classes)
        self.train_f1 = MulticlassF1Score(num_classes=n_classes, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=n_classes, average='macro')
        self.test_f1 = MulticlassF1Score(num_classes=n_classes, average='macro')


    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, out = self(x, y)
        pred = torch.argmax(out, dim=1)
        step_acc = self.train_acc(pred, y)
        step_f1 = self.train_f1(pred, y)

        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_acc', step_acc, prog_bar=True, logger=True)
        self.log('train_f1', step_f1, prog_bar=True, logger=True)

        return {
            'loss': loss,
            'step_acc': step_acc,
            'step_f1': step_f1,
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, out = self(x, y)
        pred = torch.argmax(out, dim=1)
        step_acc = self.val_acc(pred, y)
        step_f1 = self.val_f1(pred, y)

        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', step_acc, prog_bar=True, logger=True)
        self.log('val_f1', step_f1, prog_bar=True, logger=True)
        return {
            'loss': loss,
            'step_acc': step_acc,
            'step_f1': step_f1,
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, out = self(x, y)
        pred = torch.argmax(out, dim=1)
        step_acc = self.test_acc(pred, y)
        step_f1 = self.test_f1(pred, y)

        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', step_acc, prog_bar=True, logger=True)
        self.log('test_f1', step_f1, prog_bar=True, logger=True)
        return {
            'loss': loss,
            'step_acc': step_acc,
            'step_f1': step_f1,
        }

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=1e-4)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
