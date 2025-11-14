from backtesting import Strategy


class DirectionModelStrategy(Strategy):
    def init(self):
        self.lot_size = 10
        self.stop_loss = 5
        self.take_profit = 5

        self.buy_count = 0
        self.sell_count = 0

    def next(self):
        close = self.data.Close[-1]
        signal = self.data.signal[-1]
        trgt = self.data.ATRr_60[-1] / 100

        if signal == 1:
            sl_price = close - (self.stop_loss * trgt * close)
            tp_price = close + (self.take_profit * trgt * close)
            self.buy(size=self.lot_size, sl=sl_price, tp=tp_price)
            self.buy_count += 1

        elif signal == -1:
            sl_price = close + (self.stop_loss * trgt * close)
            tp_price = close - (self.take_profit * trgt * close)
            self.sell(size=self.lot_size, sl=sl_price, tp=tp_price)
            self.sell_count += 1
