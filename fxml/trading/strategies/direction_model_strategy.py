from backtesting import Strategy


class DirectionModelStrategy(Strategy):
    def init(self):
        self.lot_size = 100
        self.stop_loss = 1
        self.take_profit = 1

        self.buy_count = 0
        self.sell_count = 0

    def next(self):
        close = self.data.Close[-1]
        trgt = self.data.trgt[-1]
        side = self.data.side[-1]

        if side == 1:
            open_trades = sum(1 for trade in self.trades if trade.is_long)
            if open_trades > 0:
                return
            sl_price = close - (self.stop_loss * trgt * close)
            tp_price = close + (self.take_profit * trgt * close)
            self.buy(size=self.lot_size, sl=sl_price, tp=tp_price)
            self.buy_count += 1

        elif side == -1:
            open_trades = sum(1 for trade in self.trades if trade.is_short)
            if open_trades > 0:
                return
            sl_price = close + (self.stop_loss * trgt * close)
            tp_price = close - (self.take_profit * trgt * close)
            self.sell(size=self.lot_size, sl=sl_price, tp=tp_price)
            self.sell_count += 1
