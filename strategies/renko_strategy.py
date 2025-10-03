import logging
import asyncio
from datetime import datetime, time as dt_time, timedelta
import os
import pytz
import decimal
import json
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame, APIError
from requests.exceptions import HTTPError
from .base_strategy import BaseStrategy

load_dotenv()

class OptimizedRenkoStrategy(BaseStrategy):
    """
    An advanced, long-only Renko strategy with portfolio-level risk management to ensure
    consistent, high-quality returns by controlling overall market exposure.
    """

    def __init__(self, api_key, api_secret, base_url, symbols=None, debug=False,
                 client=None, running_event=None,
                 # Core Parameters
                 atr_period=14, entry_logic='trend_following',
                 trend_following_bricks=2, adx_period=14, adx_threshold=24,
                 trending_atr_multiplier=1.87, ranging_atr_multiplier=0.31,
                 # Trend & Volume Filters
                 use_dual_sma_filter=False, short_sma_period=20, long_sma_period=50,
                 use_volume_filter=True, volume_sma_period=20,
                 # Dynamic Position Sizing
                 trending_position_percent=0.05, ranging_position_percent=0.02,
                 # Risk Management
                 stop_loss_atr_multiplier=3.48,
                 max_concurrent_positions=10,
                 max_total_allocation_percent=0.60
                 ):
        
        super().__init__(api_key, api_secret, base_url, symbols, debug)

        # Strategy Parameters
        self.running_event = running_event
        self.atr_period = atr_period
        self.entry_logic = entry_logic
        self.trend_following_bricks = trend_following_bricks
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.trending_atr_multiplier = trending_atr_multiplier
        self.ranging_atr_multiplier = ranging_atr_multiplier
        self.use_dual_sma_filter = use_dual_sma_filter
        self.short_sma_period = short_sma_period
        self.long_sma_period = long_sma_period
        self.use_volume_filter = use_volume_filter
        self.volume_sma_period = volume_sma_period
        self.trending_position_percent = trending_position_percent
        self.ranging_position_percent = ranging_position_percent
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.max_concurrent_positions = max_concurrent_positions
        self.max_total_allocation_percent = max_total_allocation_percent
        self.loop_sleep_time_seconds = 30
        
        # State Variables
        self.last_brick_prices, self.brick_colors, self.latest_atr, self.entry_info = {}, {}, {}, {}
        self.trades_today = 0
        self.news_sentiment = {}
        self.paused_symbols = {}
        self.learned_rules = []
        self._load_learned_rules()


        if client: self.client = client
        else: self.client = self._initialize_api_client()
        if not self._is_simulation(): self._initialize_symbol_states()

    def _initialize_api_client(self):
        try:
            client = REST(key_id=self.api_key, secret_key=self.api_secret, base_url=self.base_url, api_version='v2')
            account = client.get_account()
            logging.info(f"Alpaca connection successful. Account status: {account.status}")
            return client
        except Exception as e:
            logging.error(f"Alpaca connection failed: {e}")
            from .mock_alpaca_client import MockAlpacaClient
            return MockAlpacaClient(self.api_key, self.api_secret, self.base_url)

    def _initialize_symbol_states(self):
        logging.info("Initializing states for symbols...")
        try:
            snapshots = self.client.get_snapshots(self.symbols)
            for symbol, snapshot in snapshots.items():
                if snapshot and snapshot.latest_trade:
                    self.last_brick_prices[symbol] = snapshot.latest_trade.p
                    self.brick_colors[symbol] = []
        except Exception as e:
            logging.error(f"Error initializing symbol states: {e}")

    def _is_simulation(self):
        return isinstance(getattr(self.client, 'current_tick', None), int)
        
    def _is_future_simulation(self):
        if not self._is_simulation(): return False
        return self.client.max_ticks == 252

    def _get_market_regime(self, symbol, bars_data):
        try:
            if len(bars_data) < self.adx_period: return 'ranging', 0
            adx_data = bars_data.ta.adx(length=self.adx_period)
            if adx_data is None or adx_data.empty: return 'ranging', 0
            current_adx = adx_data.iloc[-1][f'ADX_{self.adx_period}']
            if pd.isna(current_adx): return 'ranging', 0
            return ('trending', current_adx) if current_adx > self.adx_threshold else ('ranging', current_adx)
        except Exception: return 'ranging', 0

    def calculate_atr(self, symbol, bars_data):
        try:
            if len(bars_data) < self.atr_period: return None
            atr_series = bars_data.ta.atr(length=self.atr_period)
            if atr_series is None or atr_series.empty: return None
            current_atr = atr_series.iloc[-1]
            if pd.isna(current_atr): return None
            self.latest_atr[symbol] = current_atr
            return current_atr
        except Exception: return None

    def calculate_adaptive_brick_size(self, symbol, current_price, regime, bars_data):
        atr = self.calculate_atr(symbol, bars_data)
        if atr is None or atr <= 0: return current_price * 0.001
        atr_multiplier = self.trending_atr_multiplier if regime == 'trending' else self.ranging_atr_multiplier
        adaptive_size = atr * atr_multiplier
        min_size, max_size = current_price * 0.0005, current_price * 0.01
        return max(min_size, min(adaptive_size, max_size))

    def get_renko_signal(self, symbol, ohlc_data, brick_size):
        current_price = ohlc_data['close']
        if symbol not in self.last_brick_prices:
            self.last_brick_prices[symbol] = current_price
            self.brick_colors[symbol] = []
            return "hold"
        price_change = current_price - self.last_brick_prices[symbol]
        if abs(price_change) < brick_size: return "hold"
        bricks_formed = int(abs(price_change) / brick_size)
        if bricks_formed == 0: return "hold"
        new_brick_color = 1 if price_change > 0 else -1
        for _ in range(bricks_formed): self.brick_colors[symbol].append(new_brick_color)
        self.brick_colors[symbol] = self.brick_colors[symbol][-10:]
        self.last_brick_prices[symbol] += (bricks_formed * brick_size * new_brick_color)
        if self.entry_logic == 'trend_following' and len(self.brick_colors[symbol]) > self.trend_following_bricks:
            required = self.brick_colors[symbol][-self.trend_following_bricks:]
            prior = self.brick_colors[symbol][-(self.trend_following_bricks + 1)]
            if all(b == 1 for b in required) and prior == -1: return "buy"
            if all(b == -1 for b in required) and prior == 1: return "sell"
        return "hold"
    
    def _get_trend_confirmation(self, symbol, daily_bars_data):
        try:
            if len(daily_bars_data) < self.long_sma_period: return 0
            short_sma = daily_bars_data['close'].rolling(window=self.short_sma_period).mean().iloc[-1]
            long_sma = daily_bars_data['close'].rolling(window=self.long_sma_period).mean().iloc[-1]
            return 1 if short_sma > long_sma else -1
        except Exception: return 0
        
    def _has_volume_confirmation(self, symbol, bars_data):
        try:
            if 'volume' not in bars_data.columns or len(bars_data) < self.volume_sma_period: return False
            avg_volume = bars_data['volume'].rolling(window=self.volume_sma_period).mean().iloc[-2]
            current_volume = bars_data['volume'].iloc[-1]
            return current_volume > avg_volume
        except Exception:
            return False

    def calculate_position_size(self, current_price, regime, equity):
        percent = self.trending_position_percent if regime == 'trending' else self.ranging_position_percent
        size_in_dollars = equity * percent
        return int(size_in_dollars / current_price) if current_price > 0 else 0

    async def manage_open_positions(self, symbol, position, current_price, bars_data):
        if symbol not in self.entry_info or position is None: return
        atr = self.calculate_atr(symbol, bars_data)
        if not atr: return
        stop_offset = atr * self.stop_loss_atr_multiplier
        high_since_entry = max(self.entry_info[symbol].get('high_since_entry', current_price), current_price)
        self.entry_info[symbol]['high_since_entry'] = high_since_entry
        chandelier_stop = high_since_entry - stop_offset
        new_trailing_stop = max(self.entry_info[symbol].get('trailing_stop', 0), chandelier_stop)
        self.entry_info[symbol]['trailing_stop'] = new_trailing_stop
        if current_price < new_trailing_stop:
            await self.close_position(symbol)

    async def close_position(self, symbol):
        # --- THIS IS THE FIX ---
        # This new logic is safer and prevents state mismatch errors.
        try:
            position = self.client.get_position(symbol)
            qty_to_close = float(position.qty)
            side = 'sell' if qty_to_close > 0 else 'buy'
            self.client.submit_order(symbol=symbol, qty=abs(qty_to_close), side=side, type='market', time_in_force='day')
        except APIError as e:
            if "position does not exist" in str(e):
                logging.info(f"Tried to close {symbol}, but position was already gone.")
            else:
                logging.error(f"Failed to close position for {symbol}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while closing {symbol}: {e}")
        finally:
            # Always remove from internal tracking to prevent repeated closing attempts.
            if symbol in self.entry_info:
                del self.entry_info[symbol]

    async def execute_trade(self, side, symbol, current_price, regime):
        try:
            account = self.client.get_account()
            equity = float(account.equity)
            open_positions = self.client.list_positions()
            if len(open_positions) >= self.max_concurrent_positions:
                logging.warning(f"SKIPPING TRADE for {symbol}: Max concurrent positions reached.")
                return False
            portfolio_value = float(account.portfolio_value)
            if (portfolio_value / equity) >= self.max_total_allocation_percent:
                logging.warning(f"SKIPPING TRADE for {symbol}: Max capital allocation reached.")
                return False
            qty = self.calculate_position_size(current_price, regime, equity)
            if qty <= 0: return False
            if qty * current_price > float(account.buying_power): return False
            
            # --- THIS IS THE FIX ---
            # Only add to entry_info *after* the order is successfully submitted.
            order = self.client.submit_order(symbol=symbol, qty=qty, side=side, type='market', time_in_force='day')
            if order:
                self.trades_today += 1
                self.entry_info[symbol] = {'price': current_price, 'trailing_stop': 0, 'high_since_entry': current_price}
                return True
            return False
        except APIError as e:
            logging.error(f"API Error on trade execution for {symbol}: {e}")
            return False

    def is_trading_hours(self):
        now = datetime.now(pytz.timezone('US/Eastern'))
        return (dt_time(9, 30) <= now.time() <= dt_time(16, 0)) and now.weekday() < 5
        
    def _load_learned_rules(self):
        try:
            with open('learned_rules.json', 'r') as f:
                self.learned_rules = json.load(f)
            logging.info(f"Loaded {len(self.learned_rules)} learned rules.")
        except FileNotFoundError:
            logging.info("No learned_rules.json file found. Running with base strategy.")
        except Exception as e:
            logging.error(f"Error loading learned rules: {e}")

    async def update_news_sentiment(self, symbol, sentiment, headline=""):
        self.news_sentiment[symbol] = {'sentiment': sentiment, 'timestamp': datetime.now()}
        headline_lower = headline.lower()
        for rule in self.learned_rules:
            if rule['trigger_keyword'] in headline_lower:
                logging.warning(f"LEARNED RULE TRIGGERED by keyword '{rule['trigger_keyword']}' for {symbol}!")
                if symbol in self.paused_symbols and datetime.now() < self.paused_symbols[symbol]:
                    logging.critical(f"SECOND CONFIRMATION received for {symbol}. IMMEDIATE EXIT.")
                    await self.close_position(symbol)
                else:
                    duration = timedelta(minutes=rule.get('duration_minutes', 120))
                    self.paused_symbols[symbol] = datetime.now() + duration
                    logging.info(f"{symbol} entering Heightened Alert state for {duration.total_seconds()/60} mins.")
                break
        if sentiment == 'negative' and not any(rule['trigger_keyword'] in headline_lower for rule in self.learned_rules):
            try:
                if self.client.get_position(symbol):
                    logging.warning(f"General NEGATIVE NEWS for {symbol}! Immediately closing position.")
                    await self.close_position(symbol)
            except APIError: pass

    async def run_for_symbol(self, symbol, ohlc_data, position, bars_data, daily_bars_data):
        if position:
            await self.manage_open_positions(symbol, position, ohlc_data['close'], bars_data)
            return
        if symbol in self.paused_symbols and datetime.now() < self.paused_symbols[symbol]:
            logging.warning(f"BUY signal for {symbol} VETOED due to Heightened Alert state.")
            return

        regime, _ = self._get_market_regime(symbol, bars_data)
        brick_size = self.calculate_adaptive_brick_size(symbol, ohlc_data['close'], regime, bars_data)
        signal = self.get_renko_signal(symbol, ohlc_data, brick_size)
        if signal == "hold": return
            
        if signal == 'buy':
            if self.use_dual_sma_filter and self._get_trend_confirmation(symbol, daily_bars_data) == -1:
                return
            if self.use_volume_filter and not self._is_simulation() and not self._has_volume_confirmation(symbol, bars_data):
                return
            await self.execute_trade('buy', symbol, ohlc_data['close'], regime)
        
        elif signal == 'sell' and position:
            await self.close_position(symbol)
            
    async def run_strategy(self):
        logging.info("Starting Live Renko Strategy Loop (optimized with Volume Filter)")
        while self.running_event and self.running_event.is_set():
            if self.is_trading_hours():
                try:
                    open_positions = {p.symbol: p for p in self.client.list_positions()}
                    snapshots = self.client.get_snapshots(self.symbols)
                    indicator_bars_df = self.client.get_bars(self.symbols, TimeFrame.Minute, limit=self.adx_period * 3).df
                    daily_bars_df = self.client.get_bars(self.symbols, TimeFrame.Day, limit=self.long_sma_period + 5).df
                except Exception as e:
                    logging.error(f"Failed to fetch market data in batches: {e}")
                    await asyncio.sleep(self.loop_sleep_time_seconds)
                    continue

                for symbol in self.symbols:
                    s = snapshots.get(symbol)
                    if s and s.minute_bar:
                        ohlc_data = {'open': s.minute_bar.o, 'high': s.minute_bar.h, 'low': s.minute_bar.l, 'close': s.minute_bar.c, 'volume': s.minute_bar.v}
                        position = open_positions.get(symbol)
                        bars_for_symbol = indicator_bars_df.loc[symbol] if not indicator_bars_df.empty and symbol in indicator_bars_df.index.get_level_values(0) else pd.DataFrame()
                        daily_bars_for_symbol = daily_bars_df.loc[symbol] if not daily_bars_df.empty and symbol in daily_bars_df.index.get_level_values(0) else pd.DataFrame()
                        asyncio.create_task(self.run_for_symbol(symbol, ohlc_data, position, bars_for_symbol, daily_bars_for_symbol))
                
                await asyncio.sleep(1) 
            
            await asyncio.sleep(self.loop_sleep_time_seconds)

    # Helper methods for app.py
    def get_account_info(self):
        try: return self.client.get_account()
        except Exception: return None
    def get_position_qty(self, symbol):
        try: return float(self.client.get_position(symbol).qty)
        except (APIError, HTTPError): return 0.0
    def get_latest_price(self, symbol):
        try: return self.client.get_snapshots([symbol])[symbol].latest_trade.p
        except Exception: return 0

