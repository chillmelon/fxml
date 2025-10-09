import pandas as pd
import torch
from backtesting import Strategy

from models.classification.t2v_transformer_model import T2VTransformerModule

CHECKPOINT_PATH = "lightning_logs/t2v+transformer-58m-dollar-cusum_filter/version_24/checkpoints/best_checkpoint.ckpt"
model = T2VTransformerModule.load_from_checkpoint(CHECKPOINT_PATH)
model.eval()
model.to("cpu")  # or 'cuda' if you're running with GPU


class TransformerStrategy(Strategy):
    def init(self):
        self.sequence_length = 120
        self.lot_size = 0.1
        self.stop_loss_amount = 0.05
        self.take_profit_amount = 0.075

        self.buy_count = 0
        self.sell_count = 0

    def next(self):
        if len(self.data.Close) < self.sequence_length + 1:
            return  # Not enough data to predict

        # Extract the last N closes
        close_prices = self.data.Close[-self.sequence_length :]
        close_returns = (
            pd.Series(close_prices).pct_change().dropna().values.reshape(1, -1, 1)
        )
        close_tensor = torch.FloatTensor(close_returns)

        # Model prediction
        with torch.no_grad():
            _, probs = model(close_tensor)
            signal = torch.argmax(probs, dim=1).item()
            if probs[0, signal] < 0.7:
                return
        current_price = self.data.Close[-1]

        # 0 = sell, 1 = hold, 2 = buy
        if signal == 2:
            if any(trade.is_long for trade in self.trades):
                return
            sl = current_price - self.stop_loss_amount
            tp = current_price + self.take_profit_amount
            self.buy(size=self.lot_size, sl=sl, tp=tp)
            self.buy_count += 1

        if signal == 0:
            if any(trade.is_short for trade in self.trades):
                return
            sl = current_price + self.stop_loss_amount
            tp = current_price - self.take_profit_amount
            self.sell(size=self.lot_size, sl=sl, tp=tp)
            self.sell_count += 1
