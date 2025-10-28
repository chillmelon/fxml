from fxml.models.baseline_classifier.model import BaselineClassifierModule
from fxml.models.baseline_regressor.model import BaselineRegressorModule
from fxml.models.gru_regressor.model import GRURegressorModule
from fxml.models.lstm_classifier.model import LSTMClassifierModule
from fxml.models.lstm_regressor.model import LSTMRegressorModule
from fxml.models.t2v_transformer_clfr.model import T2VTransformerClassifierModule
from fxml.models.t2v_transformer_regr.model import T2VTransformerRegressorModule
from fxml.models.transformer_classifier.model import TransformerClassifierModule
from fxml.models.transformer_regressor.model import TransformerRegressorModule


def build_model(model_name, config):
    if model_name == "baseline_classifier":
        return BaselineClassifierModule(
            n_features=len(config["data"]["features"]),
            output_size=config["data"]["n_classes"],
            n_hidden=config["model"]["hidden_size"],
            dropout=config["model"]["dropout"],
            lr=config["model"]["lr"],
        )
    elif model_name == "lstm_classifier":
        return LSTMClassifierModule(
            n_features=len(config["data"]["features"]),
            output_size=config["data"]["n_classes"],
            n_hidden=config["model"]["n_hidden"],
            n_layers=config["model"]["n_layers"],
            dropout=config["model"]["dropout"],
            lr=config["model"]["lr"],
        )
    elif model_name == "transformer_classifier":
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
    elif model_name == "t2v_transformer_clfr":
        return T2VTransformerClassifierModule(
            n_timefeatures=len(config["data"]["time_features"]),
            n_features=len(config["data"]["features"]),
            output_size=config["data"]["n_classes"],
            kernel_size=config["model"]["kernel_size"],
            d_model=config["model"]["d_model"],
            nhead=config["model"]["nhead"],
            n_layers=config["model"]["n_layers"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            pool=config["model"]["pool"],
            lr=config["model"]["lr"],
        )

    elif model_name == "baseline_regressor":
        return BaselineRegressorModule(
            n_features=len(config["data"]["time_features"])
            + len(config["data"]["features"]),
            output_size=config["data"]["lookforward"],
            n_hidden=config["model"]["n_hidden"],
            dropout=config["model"]["dropout"],
            lr=config["model"]["lr"],
        )

    elif model_name == "lstm_regressor":
        return LSTMRegressorModule(
            n_features=len(config["data"]["time_features"])
            + len(config["data"]["features"]),
            output_size=config["data"]["lookforward"],
            n_hidden=config["model"]["n_hidden"],
            n_layers=config["model"]["n_layers"],
            dropout=config["model"]["dropout"],
            lr=config["model"]["lr"],
        )

    elif model_name == "gru_regressor":
        return GRURegressorModule(
            n_features=len(config["data"]["time_features"])
            + len(config["data"]["features"]),
            output_size=config["data"]["lookforward"],
            n_hidden=config["model"]["n_hidden"],
            n_layers=config["model"]["n_layers"],
            dropout=config["model"]["dropout"],
            lr=config["model"]["lr"],
        )

    elif model_name == "transformer_regressor":
        return TransformerRegressorModule(
            n_features=len(config["data"]["time_features"])
            + len(config["data"]["features"]),
            output_size=config["data"]["lookforward"],
            d_model=config["model"]["d_model"],
            nhead=config["model"]["nhead"],
            n_layers=config["model"]["n_layers"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            pool=config["model"]["pool"],
            lr=config["model"]["lr"],
        )

    elif model_name == "t2v_transformer_regr":
        return T2VTransformerRegressorModule(
            n_timefeatures=len(config["data"]["time_features"]),
            n_features=len(config["data"]["features"]),
            output_size=config["data"]["lookforward"],
            d_model=config["model"]["d_model"],
            nhead=config["model"]["nhead"],
            n_layers=config["model"]["n_layers"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            pool=config["model"]["pool"],
            lr=config["model"]["lr"],
        )

    else:
        raise ValueError(f"Model {model_name} not found")
