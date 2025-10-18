from fxml.models.baseline_classifier.model import BaselineClassifierModule
from fxml.models.lstm_classifier.model import LSTMClassifierModule
from fxml.models.lstm_regressor.model import LSTMRegressorModule
from fxml.models.transformer_classifier.model import TransformerClassifierModule
from fxml.models.transformer_regressor.model import TransformerRegressorModule


def build_model(name, config):
    if name == "baseline_classifier":
        return BaselineClassifierModule(
            n_features=len(config["data"]["features"]),
            output_size=config["data"]["n_classes"],
            hidden_size=config["model"]["hidden_size"],
            dropout=config["model"]["dropout"],
            lr=config["model"]["lr"],
        )
    elif name == "lstm_classifier":
        return LSTMClassifierModule(
            n_features=len(config["data"]["features"]),
            output_size=config["data"]["n_classes"],
            n_hidden=config["model"]["n_hidden"],
            n_layers=config["model"]["n_layers"],
            dropout=config["model"]["dropout"],
            lr=config["model"]["lr"],
        )
    elif name == "transformer_classifier":
        return TransformerClassifierModule(
            n_features=len(config["data"]["features"]),
            output_size=config["data"]["n_classes"],
            d_model=config["model"]["d_model"],
            nhead=config["model"]["nhead"],
            n_layers=config["model"]["n_layers"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            pool=config["model"]["pool"],
            lr=config["model"]["lr"],
        )
    else:
        raise ValueError(f"Model {name} not found")
