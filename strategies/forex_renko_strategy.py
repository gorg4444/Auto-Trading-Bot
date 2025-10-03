import logging
import os
import time
from datetime import datetime, time as dt_time
import pytz
from oandapyV20 import API
from oandapyV20.endpoints.accounts import AccountDetails
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.positions import PositionClose, OpenPositions
from oandapyV20.endpoints.pricing import PricingStream
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.exceptions import V20Error
import requests
from dotenv import load_dotenv

# Umgebungsvariab laden
load_dotenv()

class ForexRenkoStrategy:
    """Renko Forex Strategy für Oanda mit zwei roten Ziegeln + grün = long, zwei grüne Ziegel + rot = short."""
    
    def __init__(self, account_id, access_token, environment="practice", symbols=None, 
                 brick_size_pips=0.7, risk_per_trade=0.01, take_profit_pips=10, stop_loss_pips=5):
        """
        Initialize Forex Renko Strategy für Oanda
        
        Empfohlene Forex-Paare (Major Pairs mit engen Spreads):
        - EUR_USD (Euro/US Dollar)
        - USD_JPY (US Dollar/Japanese Yen) 
        - GBP_USD (British Pound/US Dollar)
        - USD_CHF (US Dollar/Swiss Franc)
        - AUD_USD (Australian Dollar/US Dollar)
        - USD_CAD (US Dollar/Canadian Dollar)
        - NZD_USD (New Zealand Dollar/US Dollar)
        """
        
        # Use suggested symbols if none provided
        if symbols is None:
            symbols = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'USD_CHF']
            
        self.symbols = symbols
        self.account_id = account_id
        self.environment = environment
        self.access_token = access_token
        
        # Forex trading parameters
        self.brick_size_pips = brick_size_pips  # Brick size in pips
        self.risk_per_trade = risk_per_trade  # Risk per trade as percentage of account
        self.take_profit_pips = take_profit_pips  # Take profit in pips
        self.stop_loss_pips = stop_loss_pips  # Stop loss in pips
        
        # Initialize Oanda API client
        self.client = API(access_token=access_token, environment=environment)
        
        # Renko state variables
        self.last_brick_prices = {}
        self.brick_colors = {}  # Track brick colors: 1 for green, -1 for red
        
        # Trade tracking
        self.trades_today = 0
        self.max_trades_per_day = 10
        
        # Initialize for each symbol
        for symbol in self.symbols:
            price = self.get_current_price(symbol)
            if price:
                self.last_brick_prices[symbol] = price
                self.brick_colors[symbol] = []  # Initialize empty brick history
                logging.info(f"Initialized {symbol} with price: {price}")
    
    def is_forex_trading_hours(self):
        """Check if current time is within Forex trading hours (24/5 but best liquidity during overlaps)."""
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        # Forex market is open 24/5 from Sunday 5 PM ET to Friday 5 PM ET
        # Best to avoid trading on Fridays after 2 PM ET and weekends
        if now.weekday() == 4 and now.hour >= 14:  # Friday after 2 PM ET
            return False
        if now.weekday() == 5:  # Saturday
            return False 
        if now.weekday() == 6 and now.hour < 17:  # Sunday before 5 PM ET
            return False
            
        return True

    def get_account_info(self):
        """Get Oanda account information."""
        try:
            r = AccountDetails(accountID=self.account_id)
            response = self.client.request(r)
            return response['account']
        except V20Error as e:
            logging.error(f"Error getting account info: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error getting account info: {e}")
            return None

    def get_current_price(self, instrument):
        """Get current price for a Forex instrument."""
        try:
            params = {"instruments": instrument}
            request = PricingStream(accountID=self.account_id, params=params)
            response = self.client.request(request)
            
            # Get the first response which contains current price
            for tick in response:
                if 'closeoutAsk' in tick and 'closeoutBid' in tick:
                    ask = float(tick['closeoutAsk'])
                    bid = float(tick['closeoutBid'])
                    return (ask + bid) / 2  # Return mid price
                break
                
        except V20Error as e:
            logging.error(f"Oanda API error getting price for {instrument}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error getting price for {instrument}: {e}")
        
        return None

    def get_pip_size(self, instrument):
        """Get the pip size for a Forex instrument."""
        # For most pairs, 1 pip = 0.0001, for JPY pairs it's 0.01
        if 'JPY' in instrument:
            return 0.01
        else:
            return 0.0001

    def calculate_position_size(self, instrument, current_price):
        """Calculate position size based on account equity and risk management."""
        try:
            account_info = self.get_account_info()
            if not account_info:
                return 0
                
            equity = float(account_info['NAV'])
            pip_size = self.get_pip_size(instrument)
            risk_amount = equity * self.risk_per_trade
            risk_in_pips = self.stop_loss_pips
            
            # Calculate position size: (risk amount) / (risk in pips * pip value per unit)
            pip_value_per_unit = pip_size * 1  # For standard lots, 1 pip = $10 for XXX/USD pairs
            position_size = risk_amount / (risk_in_pips * pip_value_per_unit)
            
            # Convert to units (1000 units = 0.01 lots)
            return int(position_size * 1000)  # Return in units
            
        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            return 0

    def get_open_positions(self, instrument=None):
        """Get open positions for account."""
        try:
            r = OpenPositions(accountID=self.account_id)
            response = self.client.request(r)
            
            positions = response['positions']
            if instrument:
                for position in positions:
                    if position['instrument'] == instrument:
                        return float(position['long']['units']), float(position['short']['units'])
                return 0, 0
            else:
                return positions
                
        except V20Error as e:
            logging.error(f"Error getting positions: {e}")
            return 0, 0
        except Exception as e:
            logging.error(f"Unexpected error getting positions: {e}")
            return 0, 0

    def close_position(self, instrument, long_units=0, short_units=0):
        """Close a position."""
        try:
            if long_units > 0:
                data = {"longUnits": "ALL"}
                r = PositionClose(accountID=self.account_id, instrument=instrument, data=data)
                self.client.request(r)
                logging.info(f"Closed long position for {instrument}")
                
            if short_units > 0:
                data = {"shortUnits": "ALL"}
                r = PositionClose(accountID=self.account_id, instrument=instrument, data=data)
                self.client.request(r)
                logging.info(f"Closed short position for {instrument}")
                
            return True
            
        except V20Error as e:
            logging.error(f"Error closing position for {instrument}: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error closing position for {instrument}: {e}")
            return False

    def execute_trade(self, side, instrument, current_price):
        """Execute Forex trade with Oanda."""
        if not self.is_forex_trading_hours():
            logging.info("Outside optimal Forex trading hours. Skipping trade.")
            return False
            
        if self.trades_today >= self.max_trades_per_day:
            logging.info("Daily trade limit reached.")
            return False

        try:
            # Get current positions
            long_units, short_units = self.get_open_positions(instrument)
            
            # Close opposite position first
            if (side == "buy" and short_units > 0) or (side == "sell" and long_units > 0):
                self.close_position(instrument, long_units, short_units)
                time.sleep(1)  # Wait a moment for position to close
            
            # Calculate position size
            units = self.calculate_position_size(instrument, current_price)
            if units <= 0:
                logging.warning(f"Invalid position size {units} for {instrument}. Trade skipped.")
                return False
            
            # Calculate stop loss and take profit
            pip_size = self.get_pip_size(instrument)
            
            if side == "buy":
                stop_loss_price = current_price - (self.stop_loss_pips * pip_size)
                take_profit_price = current_price + (self.take_profit_pips * pip_size)
            else:  # sell
                stop_loss_price = current_price + (self.stop_loss_pips * pip_size)
                take_profit_price = current_price - (self.take_profit_pips * pip_size)
            
            # Create order
            order_data = {
                "order": {
                    "units": str(units) if side == "buy" else str(-units),
                    "instrument": instrument,
                    "type": "MARKET",
                    "timeInForce": "FOK",
                    "positionFill": "DEFAULT",
                    "stopLossOnFill": {
                        "price": f"{stop_loss_price:.5f}"
                    },
                    "takeProfitOnFill": {
                        "price": f"{take_profit_price:.5f}"
                    }
                }
            }
            
            # Execute order
            r = OrderCreate(accountID=self.account_id, data=order_data)
            response = self.client.request(r)
            
            self.trades_today += 1
            logging.info(f"{side.upper()} order executed for {instrument}: {units} units")
            return True
            
        except V20Error as e:
            logging.error(f"Oanda API error executing {side} trade for {instrument}: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error executing {side} trade for {instrument}: {e}")
            return False

    def calculate_brick_size(self, current_price, instrument):
        """Calculate brick size based on pips."""
        pip_size = self.get_pip_size(instrument)
        return self.brick_size_pips * pip_size

    def get_renko_signal(self, instrument, current_price):
        """Generate Renko trading signal based on brick patterns."""
        if instrument not in self.last_brick_prices:
            self.last_brick_prices[instrument] = current_price
            self.brick_colors[instrument] = []
            return "hold"
        
        last_brick_price = self.last_brick_prices[instrument]
        brick_size = self.calculate_brick_size(current_price, instrument)
        
        # Check for new brick formation
        price_change = current_price - last_brick_price
        bricks_formed = int(abs(price_change) / brick_size)
        
        if bricks_formed == 0:
            return "hold"
        
        # Determine brick color and update history
        new_brick_color = 1 if price_change > 0 else -1  # 1 for green, -1 for red
        
        # Add new bricks to history
        for _ in range(bricks_formed):
            self.brick_colors[instrument].append(new_brick_color)
        
        # Keep only last 3 bricks for pattern detection
        if len(self.brick_colors[instrument]) > 3:
            self.brick_colors[instrument] = self.brick_colors[instrument][-3:]
        
        # Update last brick price
        new_brick_price = last_brick_price + (bricks_formed * brick_size * (1 if price_change > 0 else -1))
        self.last_brick_prices[instrument] = new_brick_price
        
        logging.info(f"{instrument}: {bricks_formed} new {'green' if new_brick_color == 1 else 'red'} brick(s), "
                    f"price: {current_price:.5f}, last brick: {new_brick_price:.5f}")
        
        # Check for patterns (need at least 3 bricks)
        if len(self.brick_colors[instrument]) < 3:
            return "hold"
        
        # Get last 3 bricks
        brick1, brick2, brick3 = self.brick_colors[instrument][-3:]
        
        # Pattern: Two red bricks followed by a green brick = LONG
        if brick1 == -1 and brick2 == -1 and brick3 == 1:
            logging.info(f"LONG signal for {instrument}: Two red bricks followed by green")
            return "buy"
        
        # Pattern: Two green bricks followed by a red brick = SHORT
        if brick1 == 1 and brick2 == 1 and brick3 == -1:
            logging.info(f"SHORT signal for {instrument}: Two green bricks followed by red")
            return "sell"
        
        return "hold"

    def run_for_instrument(self, instrument):
        """Run strategy for a single Forex instrument."""
        try:
            # Check if within trading hours
            if not self.is_forex_trading_hours():
                logging.debug("Outside Forex trading hours. Skipping.")
                return False
                
            logging.info(f"Checking instrument: {instrument}")
            
            current_price = self.get_current_price(instrument)
            if current_price is None:
                logging.warning(f"Could not get price for {instrument}. Skipping.")
                return False
            
            signal = self.get_renko_signal(instrument, current_price)
            long_units, short_units = self.get_open_positions(instrument)
            
            logging.info(f"{instrument}: Price={current_price:.5f}, Signal={signal}, Long={long_units}, Short={short_units}")
            
            if signal == "buy" and short_units <= 0 and long_units <= 0:
                logging.info(f"Renko BUY signal for {instrument}. Going long.")
                return self.execute_trade("buy", instrument, current_price)
            elif signal == "sell" and long_units <= 0 and short_units <= 0:
                logging.info(f"Renko SELL signal for {instrument}. Going short.")
                return self.execute_trade("sell", instrument, current_price)
            else:
                logging.debug(f"No action for {instrument}")
                return False
                
        except Exception as e:
            logging.error(f"Error processing {instrument}: {e}")
            return False

    def run_strategy(self):
        """Main method to run the strategy for all instruments."""
        logging.info("Starting Forex Renko Strategy")
        
        while True:
            try:
                for instrument in self.symbols:
                    self.run_for_instrument(instrument)
                
                # Reset daily trade count at midnight ET
                eastern = pytz.timezone('US/Eastern')
                now = datetime.now(eastern)
                if now.hour == 0 and now.minute == 0:
                    self.trades_today = 0
                
                # Wait before next iteration
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logging.info("Strategy stopped by user")
                break
            except Exception as e:
                logging.error(f"Unexpected error in main loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error

# Beispiel für die Verwendung
if __name__ == "__main__":
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("forex_renko_strategy.log"),
            logging.StreamHandler()
        ]
    )
    
    # Oanda Account Daten aus Umgebungsvariablen
    ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
    ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
    ENVIRONMENT = "practice"  # "practice" für Demo, "live" für Echtgeld
    
    # Strategy erstellen und starten
    strategy = ForexRenkoStrategy(
        account_id=ACCOUNT_ID,
        access_token=ACCESS_TOKEN,
        environment=ENVIRONMENT,
        symbols=['EUR_USD', 'USD_JPY'],  # Major pairs mit engen Spreads
        brick_size_pips=0.7,  # 0.7 Pips Brick Size
        risk_per_trade=0.01,  # 1% Risiko pro Trade
        take_profit_pips=10,  # 10 Pips Take Profit
        stop_loss_pips=5      # 5 Pips Stop Loss
    )
    
    strategy.run_strategy()