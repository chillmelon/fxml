import pandas as pd
from backtesting import Strategy

# Define Simple Moving Average (SMA) function
def SMA(data, period):
    return pd.Series(data).rolling(period).mean().values

class SimpleStrategy(Strategy):
    def init(self):
        self.fast_ma_period = 10
        self.slow_ma_period = 20

        self.lot_size = 0.1
        self.stop_loss_amount = 0.00100
        self.take_profit_amount = 0.00150

        self.buy_count = 0
        self.sell_count = 0

        # Calculate fast and slow moving averages
        self.fast_ma = self.I(SMA, self.data.Close, self.fast_ma_period)
        self.slow_ma = self.I(SMA, self.data.Close, self.slow_ma_period)

    def next(self):
        # Buy signal: Fast MA crosses above Slow MA
        if self.fast_ma[-1] > self.slow_ma[-1] and self.fast_ma[-2] <= self.slow_ma[-2]:
            open_trades = sum(1 for trade in self.trades if trade.is_long)
            if open_trades > 0:
                return

            stop_loss_price = self.data.Close[-1] - self.stop_loss_amount
            take_profit_price = self.data.Close[-1] + self.take_profit_amount
            self.buy(size=self.lot_size, sl=stop_loss_price, tp=take_profit_price)
            self.buy_count += 1

        # Sell signal: Fast MA crosses below Slow MA
        elif self.fast_ma[-1] < self.slow_ma[-1] and self.fast_ma[-2] >= self.slow_ma[-2]:
            open_trades = sum(1 for trade in self.trades if trade.is_short)
            if open_trades > 0:
                return

            stop_loss_price = self.data.Close[-1] + self.stop_loss_amount
            take_profit_price = self.data.Close[-1] - self.take_profit_amount
            self.sell(size=self.lot_size, sl=stop_loss_price, tp=take_profit_price)
            self.sell_count += 1
