import torch
from torch import nn
import lightning as pl
from torch.distributions import Normal


class ProbabilisticGRUModel(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.mu_head = nn.Linear(n_hidden, 1)
        self.log_sigma_head = nn.Linear(n_hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last_out = out[:, -1, :]  # take last timestep output
        last_out = self.dropout(last_out)
        mu = self.mu_head(last_out)                      # predicted mean
        log_sigma = self.log_sigma_head(last_out)        # predicted log-std
        return mu, log_sigma


class ProbabilisticGRURegressor(pl.LightningModule):
    def __init__(self, n_features=1, n_hidden=64, n_layers=2, dropout=0.0, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = ProbabilisticGRUModel(
            n_features=self.hparams.n_features,
            n_hidden=self.hparams.n_hidden,
            n_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout
        )

    def gaussian_nll(self, mu, log_sigma, target):
        sigma = torch.exp(log_sigma) + 1e-6  # ensure numerical stability
        dist = Normal(mu, sigma)
        nll = -dist.log_prob(target)
        return nll.mean()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu, log_sigma = self(x)
        loss = self.gaussian_nll(mu, log_sigma, y)
        sigma = torch.exp(log_sigma)

        self.log("train_mu_mean", mu.mean(), prog_bar=True)
        self.log("train_sigma_mean", sigma.mean(), prog_bar=True)
        self.log("train_nll", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mu, log_sigma = self(x)
        loss = self.gaussian_nll(mu, log_sigma, y)
        sigma = torch.exp(log_sigma)

        self.log("val_mu_mean", mu.mean(), prog_bar=True)
        self.log("val_sigma_mean", sigma.mean(), prog_bar=True)
        self.log("val_nll", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        mu, log_sigma = self(x)
        loss = self.gaussian_nll(mu, log_sigma, y)
        sigma = torch.exp(log_sigma)

        self.log("test_mu_mean", mu.mean(), prog_bar=True)
        self.log("test_sigma_mean", sigma.mean(), prog_bar=True)
        self.log("test_nll", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
