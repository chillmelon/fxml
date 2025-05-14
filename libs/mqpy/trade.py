"""Module for trading operations with MetaTrader 5 with hedging support.

Provides a Trade class for managing trading operations on hedging accounts.
This implementation specifically supports hedging accounts where multiple positions
in different directions can be open simultaneously for the same symbol.
"""

import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import MetaTrader5 as Mt5

from mqpy.logger import get_logger

# Configure logging
logger = get_logger(__name__)


class Trade:
    """Represents a trading strategy for a financial instrument with hedging support.

    Args:
        expert_name (str): The name of the expert advisor.
        version (str): The version of the expert advisor.
        symbol (str): The financial instrument symbol.
        magic_number (int): The magic number for identifying trades.
        lot (float): The number of lots to trade.
        stop_loss (float): The stop loss level.
        emergency_stop_loss (float): Emergency stop loss as a protection.
        take_profit (float): The take profit level.
        emergency_take_profit (float): Emergency take profit for gain.
        start_time (str): The time when the expert advisor can start trading.
        finishing_time (str): The time until new positions can be opened.
        ending_time (str): The time when any remaining position will be closed.
        fee (float): The average fee per trading.
    """

    def __init__(
        self,
        expert_name: str,
        version: str,
        symbol: str,
        magic_number: int,
        lot: float,
        stop_loss: float,
        emergency_stop_loss: float,
        take_profit: float,
        emergency_take_profit: float,
        start_time: str = "9:15",
        finishing_time: str = "17:30",
        ending_time: str = "17:50",
        fee: float = 0.0,
    ) -> None:
        """Initialize the Trade object.

        Returns:
            None
        """
        self.expert_name: str = expert_name
        self.version: str = version
        self.symbol: str = symbol
        self.magic_number: int = magic_number
        self.lot: float = lot
        self.stop_loss: float = stop_loss
        self.emergency_stop_loss: float = emergency_stop_loss
        self.take_profit: float = take_profit
        self.emergency_take_profit: float = emergency_take_profit
        self.start_time_hour, self.start_time_minutes = start_time.split(":")
        self.finishing_time_hour, self.finishing_time_minutes = finishing_time.split(":")
        self.ending_time_hour, self.ending_time_minutes = ending_time.split(":")
        self.fee: float = fee

        self.loss_deals: int = 0
        self.profit_deals: int = 0
        self.total_deals: int = 0
        self.balance: float = 0.0

        logger.info("Initializing the basics.")
        self.initialize()
        self.select_symbol()
        self.prepare_symbol()
        self.sl_tp_steps: float = Mt5.symbol_info(self.symbol).trade_tick_size / Mt5.symbol_info(self.symbol).point
        logger.info("Initialization successfully completed.")
        logger.info("")
        self.summary()
        logger.info("Running")
        logger.info("")

    def initialize(self) -> None:
        """Initialize the MetaTrader 5 instance.

        Returns:
            None
        """
        if not Mt5.initialize():
            logger.error("Initialization failed, check internet connection. You must have Meta Trader 5 installed.")
            Mt5.shutdown()
        else:
            logger.info(
                f"You are running the {self.expert_name} expert advisor,"
                f" version {self.version}, on symbol {self.symbol}."
            )

    def select_symbol(self) -> None:
        """Select the trading symbol.

        Returns:
            None
        """
        # Using positional arguments as the MetaTrader5 library doesn't support keywords
        Mt5.symbol_select(self.symbol, True)  # noqa: FBT003

    def prepare_symbol(self) -> None:
        """Prepare the trading symbol for opening positions.

        Returns:
            None
        """
        symbol_info = Mt5.symbol_info(self.symbol)

        if symbol_info is None:
            logger.error(f"It was not possible to find {self.symbol}")
            Mt5.shutdown()
            logger.error("Turned off")
            sys.exit(1)

        if not symbol_info.visible:
            logger.warning(f"The {self.symbol} is not visible, needed to be switched on.")
            # Using positional arguments as the MetaTrader5 library doesn't support keywords
            if not Mt5.symbol_select(self.symbol, True):  # noqa: FBT003
                logger.error(
                    f"The expert advisor {self.expert_name} failed in select the symbol {self.symbol}, turning off."
                )
                Mt5.shutdown()
                logger.error("Turned off")
                sys.exit(1)

    def is_hedging_enabled(self) -> bool:
        """Check if the account is a hedging account.

        Returns:
            bool: True if hedging is enabled, False otherwise.
        """
        account_info = Mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account information")
            return False

        # ACCOUNT_MARGIN_MODE_RETAIL_HEDGING is typically represented by 2
        # Check the actual value in your MT5 version if needed
        return account_info.margin_mode == 2  # Hedging mode

    def summary(self) -> None:
        """Print a summary of the expert advisor parameters.

        Returns:
            None
        """
        hedging_enabled = "Yes" if self.is_hedging_enabled() else "No"

        logger.info(
            f"Summary:\n"
            f"ExpertAdvisor name:              {self.expert_name}\n"
            f"ExpertAdvisor version:           {self.version}\n"
            f"Running on symbol:               {self.symbol}\n"
            f"Hedging enabled:                 {hedging_enabled}\n"
            f"MagicNumber:                     {self.magic_number}\n"
            f"Number of lot(s):                {self.lot}\n"
            f"StopLoss:                        {self.stop_loss}\n"
            f"TakeProfit:                      {self.take_profit}\n"
            f"Emergency StopLoss:              {self.emergency_stop_loss}\n"
            f"Emergency TakeProfit:            {self.emergency_take_profit}\n"
            f"Start trading time:              {self.start_time_hour}:{self.start_time_minutes}\n"
            f"Finishing trading time:          {self.finishing_time_hour}:{self.finishing_time_minutes}\n"
            f"Closing position after:          {self.ending_time_hour}:{self.ending_time_minutes}\n"
            f"Average fee per trading:         {self.fee}\n"
            f"StopLoss & TakeProfit Steps:     {self.sl_tp_steps}\n"
        )

    def statistics(self) -> None:
        """Print statistics of the expert advisor.

        Returns:
            None
        """
        logger.info(f"Total of deals: {self.total_deals}, {self.profit_deals} gain, {self.loss_deals} loss.")
        logger.info(
            f"Balance: {self.balance}, fee: {self.total_deals * self.fee}, final balance:"
            f" {self.balance - (self.total_deals * self.fee)}."
        )
        if self.total_deals != 0:
            logger.info(f"Accuracy: {round((self.profit_deals / self.total_deals) * 100, 2)}%.\n")

    def get_positions(self, position_type: Optional[int] = None) -> List[Any]:
        """Get all positions for the current symbol and magic number, optionally filtered by type.

        Args:
            position_type (Optional[int]):
                If provided, filter by position type:
                - Mt5.POSITION_TYPE_BUY (0) for buy positions
                - Mt5.POSITION_TYPE_SELL (1) for sell positions

        Returns:
            List[Any]: List of positions
        """
        positions = Mt5.positions_get(symbol=self.symbol)

        # Filter by magic number first
        positions = [pos for pos in positions if pos.magic == self.magic_number]

        # Then filter by position type if specified
        if position_type is not None:
            positions = [pos for pos in positions if pos.type == position_type]

        return positions

    def open_buy_position(self, comment: str = "") -> int:
        """Open a Buy position.

        Args:
            comment (str): A comment for the trade.

        Returns:
            int: Ticket number of the opened position, or 0 if failed
        """
        point = Mt5.symbol_info(self.symbol).point
        price = Mt5.symbol_info_tick(self.symbol).ask

        request = {
            "action": Mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot,
            "type": Mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": price - self.emergency_stop_loss * point,
            "tp": price + self.emergency_take_profit * point,
            "deviation": 5,
            "magic": self.magic_number,
            "comment": str(comment),
            "type_time": Mt5.ORDER_TIME_GTC,
            "type_filling": Mt5.ORDER_FILLING_FOK,
        }
        result = Mt5.order_send(request)

        ticket = self.process_request_result(price, result)
        return ticket

    def open_sell_position(self, comment: str = "") -> int:
        """Open a Sell position.

        Args:
            comment (str): A comment for the trade.

        Returns:
            int: Ticket number of the opened position, or 0 if failed
        """
        point = Mt5.symbol_info(self.symbol).point
        price = Mt5.symbol_info_tick(self.symbol).bid

        request = {
            "action": Mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot,
            "type": Mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": price + self.emergency_stop_loss * point,
            "tp": price - self.emergency_take_profit * point,
            "deviation": 5,
            "magic": self.magic_number,
            "comment": str(comment),
            "type_time": Mt5.ORDER_TIME_GTC,
            "type_filling": Mt5.ORDER_FILLING_FOK,
        }
        result = Mt5.order_send(request)

        ticket = self.process_request_result(price, result)
        return ticket

    def process_request_result(self, price: float, result: Any) -> int:
        """Process the result of a trading request.

        Args:
            price (float): The price of the trade.
            result (Mt5.TradeResult): The result of the trading request.

        Returns:
            int: Ticket number if successful, 0 if failed
        """
        logger.info(f"Order sent: {self.symbol}, {self.lot} lot(s), at {price}.")

        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed with error code: {result.retcode}")
            return 0

        logger.info(f"Order executed: {result.order}, ticket: {result.order}")
        return result.order

    def close_position_by_ticket(self, ticket: int, comment: str = "") -> bool:
        """Close a specific position by its ticket number.

        Args:
            ticket (int): The ticket number of the position to close.
            comment (str): A comment for the trade.

        Returns:
            bool: True if the position was closed successfully, False otherwise.
        """
        result =  Mt5.Close(symbol=self.symbol, ticket=ticket)
        if result:
            logger.info(f"Position {ticket} closed successfully")
            return True
        return False


    def open_position(self, *, should_buy: bool, should_sell: bool, comment: str = "") -> Optional[int]:
        """Open a position based on the given conditions.

        Args:
            should_buy: Whether to open a buy position.
            should_sell: Whether to open a sell position.
            comment: Optional comment for the position.

        Returns:
            Optional[int]: Ticket number of the opened position, or None if no position was opened
        """
        # Check if it's trading time
        if not self.trading_time():
            logger.info("Not within trading hours - position not opened")
            return None

        # Open position based on conditions
        if should_buy and not should_sell:
            ticket = self.open_buy_position(comment)
            if ticket:
                self.total_deals += 1
            return ticket

        if should_sell and not should_buy:
            ticket = self.open_sell_position(comment)
            if ticket:
                self.total_deals += 1
            return ticket

        return None

    def close_all_positions(self, comment: str = "") -> int:
        """Close all open positions for the current symbol and magic number.

        Args:
            comment (str): A comment for the trades.

        Returns:
            int: Number of positions closed
        """
        positions = self.get_positions()
        closed_count = 0

        for position in positions:
            Mt5.Close(symbol=self.symbol, ticket=int(position.ticket))
            if self.close_position_by_ticket(position.ticket, comment):
                closed_count += 1

        return closed_count

    def close_positions_by_type(self, position_type: int, comment: str = "") -> int:
        """Close all positions of a specific type.

        Args:
            position_type (int): The type of positions to close (Mt5.POSITION_TYPE_BUY or Mt5.POSITION_TYPE_SELL)
            comment (str): A comment for the trades.

        Returns:
            int: Number of positions closed
        """
        positions = self.get_positions(position_type)
        closed_count = 0

        for position in positions:
            if self.close_position_by_ticket(position.ticket, comment):
                closed_count += 1

        return closed_count

    def check_position_profit(self, position: Any) -> Tuple[float, bool, bool]:
        """Check if a position has reached stop loss or take profit levels.

        Args:
            position (Any): Position object from Mt5.positions_get()

        Returns:
            Tuple[float, bool, bool]: (points, take_profit_reached, stop_loss_reached)
        """
        point = Mt5.symbol_info(self.symbol).point

        # Calculate profit in points
        points = (
            position.profit
            * Mt5.symbol_info(self.symbol).trade_tick_size
            / Mt5.symbol_info(self.symbol).trade_tick_value
        ) / position.volume

        # Check if take profit or stop loss levels are reached
        take_profit_reached = points / point >= self.take_profit
        stop_loss_reached = (points / point) * -1 >= self.stop_loss

        return points, take_profit_reached, stop_loss_reached

    def stop_and_gain(self, comment: str = "") -> None:
        """Check for stop loss and take profit conditions and close positions accordingly.

        Args:
            comment (str): A comment for the trade.

        Returns:
            None
        """
        positions = self.get_positions()

        for position in positions:
            points, take_profit_reached, stop_loss_reached = self.check_position_profit(position)

            if take_profit_reached:
                self.profit_deals += 1
                self.close_position_by_ticket(position.ticket, f"TP {comment}")

                # Get the latest deal to update balance
                start_time = datetime.now(timezone.utc) - timedelta(days=1) + timedelta(hours=3)
                end_time = datetime.now(timezone.utc) + timedelta(hours=3)
                deals = Mt5.history_deals_get(start_time, end_time)

                for deal in reversed(deals):
                    if deal.position_id == position.ticket:
                        profit = deal.profit
                        logger.info(f"Take profit reached for position {position.ticket}. ({profit})\n")
                        self.balance += profit
                        break

                self.statistics()

            elif stop_loss_reached:
                self.loss_deals += 1
                self.close_position_by_ticket(position.ticket, f"SL {comment}")

                # Get the latest deal to update balance
                start_time = datetime.now(timezone.utc) - timedelta(days=1) + timedelta(hours=3)
                end_time = datetime.now(timezone.utc) + timedelta(hours=3)
                deals = Mt5.history_deals_get(start_time, end_time)

                for deal in reversed(deals):
                    if deal.position_id == position.ticket:
                        profit = deal.profit
                        logger.info(f"Stop loss reached for position {position.ticket}. ({profit})\n")
                        self.balance += profit
                        break

                self.statistics()

    def days_end(self) -> bool:
        """Check if it is the end of trading for the day.

        Returns:
            bool: True if it is the end of trading for the day, False otherwise.
        """
        now = datetime.now(timezone.utc)
        return now.hour >= int(self.ending_time_hour) and now.minute >= int(self.ending_time_minutes)

    def trading_time(self) -> bool:
        """Check if it is within the allowed trading time.

        Returns:
            bool: True if it is within the allowed trading time, False otherwise.
        """
        now = datetime.now(timezone.utc)
        if int(self.start_time_hour) < now.hour < int(self.finishing_time_hour):
            return True
        if now.hour == int(self.start_time_hour):
            return now.minute >= int(self.start_time_minutes)
        if now.hour == int(self.finishing_time_hour):
            return now.minute < int(self.finishing_time_minutes)
        return False

    def check_end_of_day(self) -> bool:
        """Check if it's the end of the trading day and close positions if necessary.

        Returns:
            bool: True if positions were closed due to end of day, False otherwise
        """
        if self.days_end():
            logger.info("It is the end of trading day.")
            logger.info("Closing all positions.")
            closed_count = self.close_all_positions("End of day")
            self.summary()
            return closed_count > 0

        return False

    def run_iteration(self, *, should_buy: bool = False, should_sell: bool = False, comment: str = "") -> None:
        """Run a single iteration of the trading strategy.

        This method performs these actions in sequence:
        1. Opens positions if conditions are met
        2. Checks existing positions for stop loss/take profit conditions
        3. Closes positions at end of day if necessary

        Args:
            should_buy (bool): Whether to open a buy position
            should_sell (bool): Whether to open a sell position
            comment (str): Comment to add to any trades

        Returns:
            None
        """
        # First check if we need to open a new position
        if should_buy or should_sell:
            self.open_position(should_buy=should_buy, should_sell=should_sell, comment=comment)

        # Then check stop loss and take profit for existing positions
        self.stop_and_gain(comment)

        # Finally check if it's end of day
        self.check_end_of_day()
