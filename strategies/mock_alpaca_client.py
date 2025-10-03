import yfinance as yf
import pandas as pd
import logging
import numpy as np

class MockAlpacaClient:
    """Simuliert den Alpaca-Client für den Debug-Modus.
    
    Verwendet yfinance, um die historischen Daten von Alpaca zu emulieren.
    """
    def __init__(self, api_key, api_secret, base_url):
        logging.warning("Running in DEBUG mode with mock Alpaca client. No real trades will be executed.")
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.positions = {}
        self.account_info = type('obj', (object,), {
            'equity': 100000.0,
            'cash': 100000.0,
            'portfolio_value': 100000.0,
            'buying_power': 100000.0
        })()
    
    def _yf_format_symbol(self, symbol):
        """Formatiert das Symbol in das von yfinance erwartete Format (z.B. 'BTC-USD')."""
        return symbol.replace('/', '-')

    def _yf_format_timeframe(self, timeframe):
        """Formatiert den Zeitrahmen von Alpaca-Format zu yfinance-Format (z.B. '5Min' -> '5m')."""
        return timeframe.replace('Min', 'm').replace('Hour', 'h').replace('Day', 'd')

    def get_crypto_bars(self, symbol, timeframe, limit=100):
        """Simuliert das Herunterladen von Krypto-Daten."""
        logging.info(f"Simulating crypto data download for {symbol}...")
        
        # Verwenden Sie yfinance als Datenquelle für den Mock-Client
        yf_symbol = self._yf_format_symbol(symbol)
        yf_timeframe = self._yf_format_timeframe(timeframe)
        
        # Calculate start date based on limit
        if 'm' in yf_timeframe:
            period = f"{limit * int(yf_timeframe.replace('m', '')) // 60 + 1}d"
        else:
            period = f"{limit + 1}d"
            
        logging.info(f"Yfinance download request: symbol='{yf_symbol}', period='{period}', interval='{yf_timeframe}'")
        data = yf.download(yf_symbol, period=period, interval=yf_timeframe, progress=False)
        
        if data.empty:
            logging.warning(f"No data found for {yf_symbol} using yfinance.")
            return type('obj', (object,), {'df': pd.DataFrame()})() # Rückgabe eines leeren DataFrames
        else:
            logging.info(f"Successfully downloaded {len(data)} rows for {yf_symbol}.")
        
        # Sicherstellen, dass die Daten die richtige Spaltenstruktur haben.
        # Konvertiere 'Adj Close' zu 'close' für Konsistenz.
        data['close'] = data['Adj Close']
        
        # Hinzufügen einer Zeitstempelspalte
        data.index.name = 'timestamp'
        
        # Sicherstellen, dass die erforderlichen Spalten vorhanden sind
        required_cols = ['close', 'open', 'high', 'low', 'volume']
        for col in required_cols:
            if col not in data.columns:
                data[col] = np.nan # Füge fehlende Spalten hinzu

        return type('obj', (object,), {'df': data})()
        
    def get_account(self):
        return self.account_info

    def get_clock(self):
        return type('obj', (object,), {'is_open': True})()
        
    def get_latest_crypto_bar(self, symbol):
        """Simuliert das Abrufen der neuesten Krypto-Bar."""
        yf_symbol = self._yf_format_symbol(symbol)
        data = yf.download(yf_symbol, period="1d", interval="1m", progress=False)
        if data.empty:
            return None
        latest_close = data['Adj Close'].iloc[-1]
        return type('obj', (object,), {'close': latest_close})()
        
    def submit_order(self, symbol, qty, side, type, time_in_force):
        logging.info(f"Submitting mock {side} order for {qty} shares of {symbol}...")
        qty_float = float(qty)
        if side == "buy":
            self.positions[symbol] = self.positions.get(symbol, 0) + qty_float
            # Aktualisiere den Kontostand
            current_price = self.get_latest_crypto_bar(symbol).close if self.get_latest_crypto_bar(symbol) else 10000
            self.account_info.equity -= qty_float * current_price
            self.account_info.cash -= qty_float * current_price
            self.account_info.buying_power = self.account_info.cash
        else:
            self.positions[symbol] = self.positions.get(symbol, 0) - qty_float
            # Aktualisiere den Kontostand
            current_price = self.get_latest_crypto_bar(symbol).close if self.get_latest_crypto_bar(symbol) else 10000
            self.account_info.equity += qty_float * current_price
            self.account_info.cash += qty_float * current_price
            self.account_info.buying_power = self.account_info.cash

        logging.info(f"Mock order for {symbol} successful. New position: {self.positions.get(symbol, 0):.6f}")
        return type('obj', (object,), {'status': 'accepted', 'symbol': symbol, 'side': side, 'qty': qty})()
        
    def get_position(self, symbol):
        qty = self.positions.get(symbol, 0)
        # Für den Mock-Client simulieren wir eine Position
        return type('obj', (object,), {'qty': qty, 'market_value': qty, 'unrealized_pl': 0, 'current_price': 100})()
    
    def list_positions(self):
        positions = []
        for symbol, qty in self.positions.items():
            if qty != 0:
                positions.append(type('obj', (object,), {
                    'symbol': symbol,
                    'qty': str(qty),
                    'market_value': qty * 100,
                    'current_price': 100,
                    'unrealized_pl': 0
                })())
        return positions