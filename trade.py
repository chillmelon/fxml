from __future__ import annotations

import logging
import MetaTrader5 as Mt5

from libs.mqpy.rate import Rates
from libs.mqpy.tick import Tick
from libs.mqpy.trade import Trade

import numpy as np
import pandas as pd
from models.gru_model import GRUModule
import torch


# Setting Parameters
CHECKPOINT_PATH = r'lightning_logs\prob_gru\version_6\checkpoints\best_checkpoint.ckpt'
WINDOW_SIZE = 30


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Loading model at global scope
model = GRUModule.load_from_checkpoint(CHECKPOINT_PATH)
model.to('cpu')
model.eval()

def gru_strategy(close_prices, conf_threshold=0.5):
    """
    Predicts using a GRU model
    """
    class_map = {0: 'sell', 1: 'hold', 2: 'buy'}
    close_returns = np.diff(close_prices) / close_prices[:-1]
    with torch.no_grad():
        _, probs = model(torch.FloatTensor(close_returns.reshape(1, -1, 1)))

        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = probs[0, pred_class].item()

        if confidence >= conf_threshold:
            signal = class_map[pred_class]
        else:
            signal = 'hold'  # fallback to 'hold' if not confident

        return signal


    # calculate returns


def main() -> None:
    # Initialize the trading strategy
    trade = Trade(
        expert_name="Crazy Buy",
        version="1.0",
        symbol="USDJPY.sml",
        magic_number=5678,
        lot=0.1,
        stop_loss=25,
        emergency_stop_loss=300,
        take_profit=25,
        emergency_take_profit=300,
        start_time="00:00",
        finishing_time="23:59",
        ending_time="23:59",
        fee=0.5,
    )

    logger.info(f"Starting crazy buying strategy on {trade.symbol}")

    # Strategy parameters
    prev_tick_time = 0
    prev_bar_time = 0
    prev_signal = 'hold'


    try:
        while True:
            # Prepare the symbol for trading
            trade.prepare_symbol()

            # Fetch tick and rates data
            current_tick = Tick(trade.symbol)

            has_new_tick = current_tick.time_msc != prev_tick_time

            if has_new_tick:
                historical_rates = Rates(trade.symbol, 1, 0, WINDOW_SIZE + 1)  # Get extra data for reliability
                has_enough_data = len(historical_rates.close) >= WINDOW_SIZE + 1

                signal = 'hold'
                should_buy, should_sell = False, False
                if (has_enough_data):
                    current_bar_time = historical_rates.time[-1]
                    if current_bar_time == prev_bar_time:
                        signal = gru_strategy(historical_rates.close)
                        if signal == prev_signal:
                            continue
                        print(signal)
                        should_buy, should_sell = (signal == 'buy'), (signal == 'sell')

                    else:
                        # new bar formed
                        signal = gru_strategy(historical_rates.close)
                        print(signal)
                        should_buy, should_sell = (signal == 'buy'), (signal == 'sell')
                        prev_bar_time = current_bar_time  # Update bar time

                    prev_signal = signal

                # Execute trading positions based on signals
                if trade.trading_time():  # Only trade during allowed hours
                    trade.open_position(
                        should_buy=should_buy,
                        should_sell=should_sell,
                        comment="Crazy"
                    )

                # Update trading statistics periodically
                trade.statistics()

            prev_tick_time = current_tick.time_msc

            # Check if it's the end of the trading day
            if trade.days_end():
                trade.close_position("End of the trading day reached.")
                break

    except KeyboardInterrupt:
        logger.info("Strategy execution interrupted by user.")
        trade.close_position("User interrupted the strategy.")
    except Exception:
        logger.exception("Error in strategy execution")
    finally:
        logger.info("Finishing the program.")


if __name__ == "__main__":
    main()
