import yfinance as yf
import pandas as pd
import time
import logging
from alpaca_trade_api.rest import REST, APIError
import numpy as np

from .base_strategy import BaseStrategy

class EMACrossoverStrategy(BaseStrategy):
    """Implementiert eine EMA-Crossover-Handelsstrategie für Kryptowährungen."""
    
    def __init__(self, api_key, api_secret, base_url, symbols=['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'LTC/USD'], 
                 base_qty=0.001, debug=False, timeframe="5Min", short_ema=3, long_ema=5, 
                 risk_per_trade=0.02, max_portfolio_risk=0.1):
        
        super().__init__(api_key, api_secret, base_url, symbols, debug)
        
        self.base_qty = base_qty
        self.timeframe = timeframe
        self.short_ema = short_ema
        self.long_ema = long_ema
        self.risk_per_trade = risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.trades_today = 0
        self.max_trades_per_day = 50
        self.last_signal = {symbol: "hold" for symbol in symbols}
        self.consecutive_signals = {symbol: 0 for symbol in symbols}
        
        if self.debug:
            from .mock_alpaca_client import MockAlpacaClient
            self.client = MockAlpacaClient(self.api_key, self.api_secret, self.base_url)
        else:
            self.client = REST(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url
            )
            logging.info("Running in LIVE mode. Real trades will be executed.")
            
        logging.info("Starting the aggressive EMA Crossover strategy loop...")

    def _format_symbol(self, symbol):
        """Formatiert das Symbol in das von der Alpaca API erwartete Format (z.B. 'BTC/USD')."""
        if '/' in symbol:
            return symbol
        if symbol.endswith('USD'):
            return f"{symbol[:-3]}/{symbol[-3:]}"
        return symbol

    def get_account_info(self):
        """Ruft Kontoinformationen ab."""
        try:
            account = self.client.get_account()
            logging.info(f"Kontostatus: Equity={account.equity}, Cash={account.cash}, Buying Power={account.buying_power}")
            return account
        except APIError as e:
            logging.error(f"API-Fehler beim Abrufen der Kontoinformationen: {e.args[0]}")
            return None
        except Exception as e:
            logging.error(f"Unerwarteter Fehler beim Abrufen der Kontoinformationen: {e}")
            return None

    def get_position_qty(self, symbol):
        """Ruft die Positionsmenge für ein Symbol ab."""
        formatted_symbol = self._format_symbol(symbol)
        try:
            position = self.client.get_position(formatted_symbol)
            qty = float(position.qty)
            return qty
        except APIError as e:
            # Alpaca gibt einen 404-Fehler zurück, wenn keine Position existiert
            if hasattr(e, 'status_code') and e.status_code == 404:
                return 0.0
            else:
                logging.error(f"API-Fehler beim Abrufen der Position für {formatted_symbol}: {e}")
                return 0.0
        except Exception as e:
            logging.error(f"Unerwarteter Fehler beim Abrufen der Position für {formatted_symbol}: {e}")
            return 0.0

    def get_latest_price(self, symbol):
        """Ruft den aktuellen Preis für ein Symbol ab."""
        formatted_symbol = self._format_symbol(symbol)
        try:
            bar = self.client.get_latest_crypto_bar(formatted_symbol)
            if bar:
                return bar.close
            return None
        except Exception as e:
            logging.error(f"Fehler beim Abrufen des aktuellen Preises für {formatted_symbol}: {e}")
            return None

    def get_crypto_data(self, symbol):
        """Ruft historische Krypto-Daten ab."""
        formatted_symbol = self._format_symbol(symbol)
        try:
            # Wir holen 100 Bars historischer Daten, um eine stabile EMA-Berechnung zu gewährleisten.
            data = self.client.get_crypto_bars(
                formatted_symbol,
                timeframe=self.timeframe,
                limit=100
            ).df
            
            logging.info(f"Abgerufene Daten für {formatted_symbol}: {len(data)} Zeilen gefunden.")
            
            # Wichtiger Schritt: Die Indexbezeichnung ändern, um sie konsistent zu machen
            data.index.name = 'timestamp'
            
            return data

        except Exception as e:
            logging.error(f"Fehler beim Abrufen von Krypto-Daten für {formatted_symbol}: {e}")
            return None

    def calculate_ema(self, df):
        """Berechnet die EMAs für den DataFrame."""
        if df.empty or 'close' not in df.columns:
            return pd.DataFrame()
            
        df['short_ema'] = df['close'].ewm(span=self.short_ema, adjust=False).mean()
        df['long_ema'] = df['close'].ewm(span=self.long_ema, adjust=False).mean()
        
        # Zusätzliche Indikatoren für aggressiveres Trading
        df['ema_ratio'] = df['short_ema'] / df['long_ema']
        df['price_vs_short'] = df['close'] / df['short_ema']
        
        return df

    def calculate_dynamic_position_size(self, symbol, current_price, account_equity):
        """Berechnet die Positionsgröße basierend auf Risikomanagement."""
        try:
            # Einfachere Risikoberechnung: Kaufe für einen festen Betrag (z.B. 1% des Kapitals)
            # oder basierend auf der Kaufkraft
            buy_value = account_equity * self.risk_per_trade * 2 # Versuchen, 4% des Kapitals pro Trade zu riskieren
            account = self.get_account_info()
            
            if account and hasattr(account, 'buying_power'):
                buying_power = float(account.buying_power)
                position_value = min(buy_value, buying_power * 0.5) # Nicht mehr als 50% der Kaufkraft
            else:
                position_value = buy_value
            
            # Berechne die Anzahl der Einheiten basierend auf dem aktuellen Preis
            if current_price == 0:
                logging.warning(f"Aktueller Preis für {symbol} ist 0. Verwende Basis-Menge.")
                return self.base_qty
            
            qty = position_value / current_price
            
            # Runde auf eine sinnvolle Dezimalstelle
            qty = round(qty, 6) # Alpaca unterstützt bis zu 6 Dezimalstellen für Krypto
                
            logging.info(f"Dynamische Positionsgröße für {symbol}: {qty} Einheiten (Wert: ~${position_value:.2f})")
            return max(qty, self.base_qty)  # Mindestens die Basis-Quantität
            
        except Exception as e:
            logging.error(f"Fehler bei der Berechnung der Positionsgröße: {e}")
            return self.base_qty

    def get_trading_signal(self, df, symbol):
        """Bestimmt das Handelssignal basierend auf dem EMA-Crossover und zusätzlichen Faktoren."""
        if df.empty or len(df) < max(self.short_ema, self.long_ema) + 5:
            return "no_signal", 0

        # Die letzten beiden Bars für das Crossover-Signal
        last = df.iloc[-1]
        second_last = df.iloc[-2]

        signal = "hold"
        strength = 0
        
        # Kaufsignal: Kurze EMA kreuzt lange EMA von unten
        if (second_last['short_ema'] < second_last['long_ema'] and last['short_ema'] > last['long_ema']):
            signal = "buy"
            strength += 1
            # Zusätzliche bullish Faktoren
            if last['ema_ratio'] > 1.005:  # Stärkeres Crossover
                strength += 1
            if last['price_vs_short'] > 1.01:  # Preis über kurzer EMA
                strength += 1

        # Verkaufssignal: Kurze EMA kreuzt lange EMA von oben
        elif (second_last['short_ema'] > second_last['long_ema'] and last['short_ema'] < last['long_ema']):
            signal = "sell"
            strength += 1
            # Zusätzliche bearish Faktoren
            if last['ema_ratio'] < 0.995:  # Stärkeres Crossover
                strength += 1
            if last['price_vs_short'] < 0.99:  # Preis unter kurzer EMA
                strength += 1

        return signal, strength

    def execute_trade(self, side, symbol, strength, current_price, current_position_qty, account_equity):
        """Führt einen Kauf- oder Verkaufstrade aus."""
        logging.info(f"Versuche, einen '{side}' Trade (Stärke: {strength}) für {symbol} auszuführen.")
        
        formatted_symbol = self._format_symbol(symbol)

        try:
            # Berechne dynamische Positionsgröße basierend auf Signalstärke
            base_qty = self.calculate_dynamic_position_size(formatted_symbol, current_price, account_equity)
            multiplier = 1.0 + (strength * 0.5)
            qty = base_qty * multiplier

            if side == "buy":
                # Nur kaufen, wenn wir keine offene Position haben oder die Position erhöhen wollen
                if current_position_qty <= 0:
                    logging.info(f"Entscheidung: Kaufsignal für {formatted_symbol} erkannt. Sende Kaufauftrag für {qty:.6f} Einheiten...")
                    self.client.submit_order(
                        symbol=formatted_symbol,
                        qty=qty,
                        side="buy",
                        type="market",
                        time_in_force="gtc"
                    )
                    logging.info(f"Kaufauftrag für {qty:.6f} Einheiten von {formatted_symbol} erfolgreich gesendet.")
                    self.trades_today += 1
                    return True
                else:
                    logging.info(f"Entscheidung: Kaufsignal erkannt, aber wir halten bereits {current_position_qty:.6f} Einheiten. Trade übersprungen.")
                    return False
            
            elif side == "sell" and current_position_qty > 0:
                # Verkaufe die gesamte Position
                logging.info(f"Entscheidung: Verkaufssignal für {formatted_symbol} erkannt. Sende Verkaufsauftrag für {current_position_qty:.6f} Einheiten...")
                self.client.submit_order(
                    symbol=formatted_symbol,
                    qty=current_position_qty,
                    side="sell",
                    type="market",
                    time_in_force="gtc"
                )
                logging.info(f"Verkaufsauftrag für {current_position_qty:.6f} Einheiten von {formatted_symbol} erfolgreich gesendet.")
                self.trades_today += 1
                return True
            else:
                logging.info(f"Entscheidung: Keine Aktion für {formatted_symbol}. Aktuelles Signal ist '{side}', aktuelle Position ist {current_position_qty:.6f} Einheiten.")
                return False

        except APIError as e:
            logging.error(f"API-Fehler bei der Ausführung des Trades für {formatted_symbol}: {e}")
            return False
        except Exception as e:
            logging.error(f"Unerwarteter Fehler bei der Ausführung des Trades für {formatted_symbol}: {e}")
            return False

    def run_for_symbol(self, symbol):
        """Führt die Strategie für ein einzelnes Symbol aus."""
        try:
            logging.info(f"=== Starte die Verarbeitung für Symbol: {symbol} ===")
            
            # Prüfe ob maximale Trades erreicht
            if self.trades_today >= self.max_trades_per_day:
                logging.warning(f"Maximale Trades pro Tag ({self.max_trades_per_day}) erreicht. Überspringe {symbol}.")
                return
            
            # 1. Daten abrufen und EMA berechnen
            data = self.get_crypto_data(symbol)
            if data is None or data.empty or len(data) < max(self.short_ema, self.long_ema) + 5:
                logging.warning(f"Keine ausreichenden Daten für {symbol} gefunden. Überspringe dieses Symbol.")
                return

            df = self.calculate_ema(data)

            # 2. Handelssignal bestimmen
            signal, strength = self.get_trading_signal(df, symbol)

            # 3. Zusätzliche Überprüfungen vor der Handelsausführung
            current_position_qty = self.get_position_qty(symbol)
            current_price = self.get_latest_price(symbol)
            account = self.get_account_info()

            if not current_price:
                logging.error(f"Konnte aktuellen Preis für {symbol} nicht abrufen. Trade wird nicht ausgeführt.")
                return

            account_equity = float(account.equity) if account else 100000

            logging.info(f"Signal für {symbol} ist '{signal}' mit Stärke {strength}. Aktuelle Position: {current_position_qty:.6f}")
            
            # 4. Trade ausführen basierend auf dem Signal
            if signal != "hold" and signal != "no_signal":
                return self.execute_trade(signal, symbol, strength, current_price, current_position_qty, account_equity)
            else:
                logging.info(f"Halte die Position für {symbol}. Aktuelle Position: {current_position_qty:.6f} Einheiten.")
                return False
            
            logging.info(f"=== Verarbeitung für Symbol {symbol} abgeschlossen. ===\n")

        except Exception as e:
            logging.error(f"Ein kritischer Fehler ist bei der Verarbeitung von {symbol} aufgetreten: {e}", exc_info=True)