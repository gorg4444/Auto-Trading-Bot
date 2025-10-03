import yfinance as yf
import pandas as pd
import numpy as np
import logging
from alpaca_trade_api.rest import APIError
import csv
import os
import json
from datetime import datetime, timedelta
from strategies.news_processor import NewsProcessor

class MockData:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)

class SimulatedAlpacaClient:
    """A mock Alpaca API client that simulates trading, logs transactions, and fetches rich historical news."""
    def __init__(self, symbols, initial_equity=100000.00, silent_logging=False):
        self.symbols = symbols
        self.silent_logging = silent_logging
        self.historical_data = {}
        self.current_tick = 0
        self.max_ticks = 0
        self.current_date = None
        self.account = MockData(equity=initial_equity, cash=initial_equity, portfolio_value=0.0, buying_power=initial_equity)
        self.positions = {}
        
        self.trade_log_file = 'simulation_trades.csv'
        self.equity_log_file = 'simulation_equity_log.csv'
        self.news_log_file = 'simulation_news_log.json'
        
        self._initialize_trade_log()
        self._initialize_equity_log()

    def _initialize_trade_log(self):
        try:
            with open(self.trade_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Symbol', 'Side', 'Qty', 'Price', 'Reason'])
            if not self.silent_logging: logging.info(f"Trade log initialized at {os.path.abspath(self.trade_log_file)}")
        except Exception as e:
            logging.error(f"Failed to initialize trade log: {e}")

    def _initialize_equity_log(self):
        try:
            with open(self.equity_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Equity', 'Cash', 'PortfolioValue'])
        except Exception as e:
            logging.error(f"Failed to initialize equity log: {e}")

    def _log_equity(self):
        try:
            with open(self.equity_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.current_date.strftime('%Y-%m-%d'), f"{self.account.equity:.2f}", f"{self.account.cash:.2f}", f"{self.account.portfolio_value:.2f}"])
        except Exception as e:
            logging.error(f"Failed to write to equity log: {e}")

    def _log_trade(self, symbol, side, qty, price, reason="Signal"):
        try:
            with open(self.trade_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                timestamp = self.current_date.strftime('%Y-%m-%d')
                writer.writerow([timestamp, symbol, side.upper(), qty, f"{price:.2f}", reason])
        except Exception as e:
            logging.error(f"Failed to write to trade log: {e}")

    def _fetch_and_save_historical_news(self, start, end):
        news_processor = NewsProcessor()
        news_data = [] 
        try:
            news_data = news_processor.get_historical_news(self.symbols, start, end)
        except Exception as e:
            logging.error(f"Failed to fetch historical news: {e}")
        finally:
            with open(self.news_log_file, 'w') as f:
                json.dump(news_data, f)
            logging.info(f"Saved {len(news_data)} historical news articles to {self.news_log_file}")

    def load_historical_data(self, period=None, start=None, end=None, interval="1d"):
        if not self.silent_logging:
            logging.info(f"Downloading historical data for {self.symbols}...")
        try:
            data = yf.download(self.symbols, period=period, start=start, end=end, interval=interval, group_by='ticker', progress=not self.silent_logging)
            valid_symbols = []
            for symbol in self.symbols:
                symbol_data = data.get(symbol)
                if symbol_data is not None and not symbol_data.dropna().empty:
                    self.historical_data[symbol] = symbol_data.dropna()
                    valid_symbols.append(symbol)
                else:
                    logging.warning(f"No historical data for {symbol}, it will be excluded from this run.")
            
            self.symbols = valid_symbols
            if self.symbols:
                self.max_ticks = len(next(iter(self.historical_data.values())))
                self.current_date = next(iter(self.historical_data.values())).index[0]
                sim_start_date = self.historical_data[self.symbols[0]].index[0]
                sim_end_date = self.historical_data[self.symbols[0]].index[-1]
                self._fetch_and_save_historical_news(sim_start_date, sim_end_date)
            if not self.silent_logging:
                logging.info(f"Loaded {self.max_ticks} data points for simulation.")
            return True
        except Exception as e:
            logging.error(f"Failed to download historical data: {e}")
            return False

    def generate_future_data(self):
        if not self.silent_logging: logging.info("Generating future year data based on last 20 years...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=20*365)
        try:
            hist_data = yf.download(self.symbols, start=start_date, end=end_date, group_by='ticker', progress=not self.silent_logging)
            future_dates = pd.to_datetime(pd.date_range(end_date, periods=252, freq='B'))
            valid_symbols = []
            for symbol in self.symbols:
                s_data = hist_data.get(symbol)
                if s_data is None or s_data.dropna().empty:
                    logging.warning(f"Not enough historical data for {symbol} to generate future. Excluding.")
                    continue
                s_data = s_data.dropna()
                log_returns = np.log(s_data['Close'] / s_data['Close'].shift(1)).dropna()
                drift = log_returns.mean()
                volatility = log_returns.std()
                last_price = s_data['Close'].iloc[-1]
                daily_returns = np.exp(drift + volatility * np.random.standard_normal(252))
                future_prices = last_price * daily_returns.cumprod()
                future_df = pd.DataFrame(index=future_dates, data={'Close': future_prices})
                future_df['Open'] = future_df['Close'].shift(1).fillna(last_price)
                price_diff = future_df['Close'] - future_df['Open']
                future_df['High'] = future_df[['Open', 'Close']].max(axis=1) + abs(price_diff * np.random.uniform(0, 0.2, 252))
                future_df['Low'] = future_df[['Open', 'Close']].min(axis=1) - abs(price_diff * np.random.uniform(0, 0.2, 252))
                future_df['Volume'] = np.random.randint(100000, 10000000, size=252)
                self.historical_data[symbol] = future_df
                valid_symbols.append(symbol)
            self.symbols = valid_symbols
            if self.symbols:
                self.max_ticks = 252
                self.current_date = self.historical_data[self.symbols[0]].index[0]
            logging.info(f"Generated a 252-day future simulation.")
            return True
        except Exception as e:
            logging.error(f"Failed to generate future data: {e}")
            return False

    def tick(self):
        if self.current_tick >= self.max_ticks - 1: return False
        self.current_tick += 1
        new_portfolio_value = 0.0
        for symbol, pos in self.positions.items():
            try:
                current_price = self.historical_data[symbol]['Close'].iloc[self.current_tick]
                self.current_date = self.historical_data[symbol].index[self.current_tick]
                pos.market_value = float(pos.qty) * current_price
                new_portfolio_value += pos.market_value
            except IndexError: pass
        self.account.portfolio_value = new_portfolio_value
        self.account.equity = self.account.cash + self.account.portfolio_value
        self.account.buying_power = self.account.cash
        self._log_equity()
        return True

    def get_account(self): return self.account

    def get_snapshots(self, symbols):
        snapshots = {}
        for symbol in symbols:
            try:
                data = self.historical_data[symbol].iloc[self.current_tick]
                snapshots[symbol] = MockData(
                    latest_trade=MockData(p=data['Close']),
                    minute_bar=MockData(o=data['Open'], h=data['High'], l=data['Low'], c=data['Close'], v=data.get('Volume', 0))
                )
            except (KeyError, IndexError): snapshots[symbol] = None
        return snapshots

    def get_bars(self, symbol, timeframe, limit=100, **kwargs):
        start = max(0, self.current_tick - limit + 1)
        end = self.current_tick + 1
        bars_df = self.historical_data[symbol].iloc[start:end].copy()
        bars_df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        return MockData(df=bars_df)

    def get_position(self, symbol):
        if symbol in self.positions: return self.positions[symbol]
        raise APIError({'code': 40410000, 'message': 'position does not exist'})

    def list_positions(self):
        return list(self.positions.values())

    def close_position(self, symbol):
        if symbol in self.positions:
            position = self.positions[symbol]
            self.submit_order(symbol, float(position.qty), 'sell', 'market', 'day', reason="Exit Signal/Stop")
        else:
            raise APIError({'code': 40410000, 'message': 'position does not exist'})

    def submit_order(self, symbol, qty, side, type, time_in_force, **kwargs):
        price = self.historical_data[symbol]['Close'].iloc[self.current_tick]
        trade_value = float(qty) * price
        
        if side == 'buy':
            if trade_value > self.account.buying_power:
                raise APIError({'code': 40310000, 'message': 'insufficient buying power'})
            self._log_trade(symbol, side, qty, price)
            self.account.cash -= trade_value
            if symbol in self.positions:
                pos = self.positions[symbol]
                new_cost = (float(pos.avg_entry_price) * float(pos.qty)) + trade_value
                new_qty = float(pos.qty) + float(qty)
                pos.avg_entry_price = new_cost / new_qty
                pos.qty = new_qty
                pos.qty_available = new_qty
            else:
                self.positions[symbol] = MockData(symbol=symbol, qty=qty, qty_available=qty, side='long', avg_entry_price=price)
        elif side == 'sell':
            if symbol not in self.positions or float(self.positions[symbol].qty) < float(qty):
                raise APIError({'code': 40310000, 'message': f'insufficient qty to sell {qty} of {symbol}'})
            self._log_trade(symbol, side, qty, price, reason=kwargs.get('reason', 'Signal'))
            self.account.cash += trade_value
            new_qty = float(self.positions[symbol].qty) - float(qty)
            self.positions[symbol].qty = new_qty
            self.positions[symbol].qty_available = new_qty
            if self.positions[symbol].qty == 0: del self.positions[symbol]
        
        self.account.buying_power = self.account.cash
        return MockData(id='sim_order', status='filled')