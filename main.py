import sys
import logging
import threading
import asyncio
import pandas as pd
from datetime import datetime
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QComboBox, QLabel, QTableWidget, QTableWidgetItem, 
                             QTextEdit, QTabWidget, QScrollArea)
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPainter, QColor

# --- Import Your Bot's Brain ---
from strategies.renko_strategy import OptimizedRenkoStrategy
from strategies.market_simulator import SimulatedAlpacaClient
from strategies.optimizer import OptimizationEngine
from strategies.news_processor import NewsProcessor
from strategies.analyzer import PerformanceAnalyzer

# --- Alpaca Stream Import ---
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.rest import TimeFrame, APIError
from requests.exceptions import HTTPError

# --- Global Configuration ---
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

SYMBOLS = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'EEM', 'EFA', 'EWJ', 'EWZ', 'FXI', 'INDA', 'XLK', 
    'XLF', 'XLV', 'XLE', 'XLI', 'GLD', 'SLV', 'USO', 'VXX', 'AAPL', 'MSFT', 
    'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'BA', 'DIS', 'NKE'
]

STRATEGY_PRESETS = {
    "aggressive_momentum": {
        "label": "Aggressive (Simulated ~19% Return)",
        "params": {"adx_threshold": 24, "trending_atr_multiplier": 1.87, "ranging_atr_multiplier": 0.31, "stop_loss_atr_multiplier": 3.48, "use_dual_sma_filter": False, "use_volume_filter": False, "max_concurrent_positions": 20, "max_total_allocation_percent": 0.80}
    },
    "robust_manager": {
        "label": "Robust (Simulated ~9% Return)",
        "params": {"adx_threshold": 26, "trending_atr_multiplier": 1.0, "ranging_atr_multiplier": 0.5, "stop_loss_atr_multiplier": 2.5, "use_dual_sma_filter": True, "use_volume_filter": True, "max_concurrent_positions": 10, "max_total_allocation_percent": 0.60}
    }
}

# --- Thread-Safe Logging and Communication ---
class QTextEditLogger(logging.Handler, QObject):
    appendPlainText = pyqtSignal(str)
    def __init__(self, parent):
        super().__init__()
        QObject.__init__(self)
        self.widget = QTextEdit(parent)
        self.widget.setReadOnly(True)
        self.appendPlainText.connect(self.widget.append)

    def emit(self, record):
        msg = self.format(record)
        self.appendPlainText.emit(msg)

class BackendSignals(QObject):
    account_update = pyqtSignal(dict)
    portfolio_update = pyqtSignal(list)
    strategy_update = pyqtSignal(dict)
    news_update = pyqtSignal(dict)
    optimization_update = pyqtSignal(dict)
    simulation_status = pyqtSignal(dict)
    analysis_result = pyqtSignal(dict)

# --- Main Application Window ---
class TradingBotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Self-Optimizing Renko Bot")
        self.setGeometry(100, 100, 1800, 1000)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1a1a2e; color: #e0e0e0; }
            QLabel { font-size: 14px; }
            QPushButton { background-color: #4a4a6a; border: 1px solid #6a6a8a; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #6a6a8a; }
            QComboBox, QLineEdit, QTextEdit { background-color: #2a2a44; border: 1px solid #3c3c5c; padding: 5px; border-radius: 4px; }
            QTableWidget { background-color: #2a2a44; border: 1px solid #3c3c5c; gridline-color: #3c3c5c; }
            QHeaderView::section { background-color: #3c3c5c; padding: 4px; border: 1px solid #2a2a44; }
            QTabWidget::pane { border: 1px solid #3c3c5c; }
            QTabBar::tab { background: #2a2a44; padding: 10px; border: 1px solid #3c3c5c; }
            QTabBar::tab:selected { background: #3c3c5c; }
            QScrollArea { border: none; }
        """)
        
        self.active_strategy = None
        self.strategy_thread = None
        self.strategy_running = threading.Event()
        self.is_simulation = False
        self.news_processor = NewsProcessor()
        self.signals = BackendSignals()
        self.news_stream = None
        
        self._setup_ui()
        self._connect_signals()
        self._start_news_stream()

    def _setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        header = QLabel("Self-Optimizing Renko Bot")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(header)
        
        top_grid = QHBoxLayout()
        main_layout.addLayout(top_grid)
        
        self.account_panel = self._create_account_panel()
        self.controls_panel = self._create_controls_panel()
        self.simulation_panel = self._create_simulation_panel()
        self.optimizer_panel = self._create_optimizer_panel()

        top_grid.addWidget(self.account_panel, 1)
        top_grid.addWidget(self.controls_panel, 1)
        top_grid.addWidget(self.simulation_panel, 1)
        top_grid.addWidget(self.optimizer_panel, 2)
        
        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout, stretch=2)

        tabs = QTabWidget()
        bottom_layout.addWidget(tabs, stretch=3)
        
        self.portfolio_panel = self._create_portfolio_panel()
        self.news_panel = self._create_news_panel()
        self.log_panel = self._create_log_panel()
        self.analysis_panel = self._create_analysis_panel()
        
        tabs.addTab(self.portfolio_panel, "Portfolio")
        tabs.addTab(self.news_panel, "Live News")
        tabs.addTab(self.log_panel, "Logs")
        tabs.addTab(self.analysis_panel, "Performance Analysis")

        self.chart_view = self._create_chart_panel()
        bottom_layout.addWidget(self.chart_view, stretch=2)

    def _connect_signals(self):
        self.start_live_button.clicked.connect(lambda: self.start_strategy(is_simulation=False))
        self.stop_button.clicked.connect(self.stop_strategy)
        self.run_sim_button.clicked.connect(self.run_full_simulation)
        self.analyze_button.clicked.connect(self.analyze_simulation)
        self.optimize_button.clicked.connect(self.start_optimization)

        self.signals.account_update.connect(self.update_account_display)
        self.signals.portfolio_update.connect(self.update_portfolio_display)
        self.signals.strategy_update.connect(self.update_strategy_status)
        self.signals.news_update.connect(self.update_news_feed)
        self.signals.optimization_update.connect(self.update_optimizer_status)
        self.signals.simulation_status.connect(self.update_simulation_status)
        self.signals.analysis_result.connect(self.update_analysis_display)

    def _create_account_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.equity_label = QLabel("Equity: $0.00")
        self.cash_label = QLabel("Cash: $0.00")
        self.value_label = QLabel("Value: $0.00")
        self.date_label = QLabel("Date: N/A")
        layout.addWidget(self.equity_label)
        layout.addWidget(self.cash_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.date_label)
        return panel

    def _create_controls_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Select Strategy Preset:"))
        self.preset_select = QComboBox()
        for name, details in STRATEGY_PRESETS.items():
            self.preset_select.addItem(details["label"], name)
        layout.addWidget(self.preset_select)
        
        self.start_live_button = QPushButton("Start Live")
        self.stop_button = QPushButton("Stop")
        layout.addWidget(self.start_live_button)
        layout.addWidget(self.stop_button)
        self.strategy_status_label = QLabel("Not Running")
        self.strategy_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.strategy_status_label)
        return panel

    def _create_simulation_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Select Year to Simulate:"))
        self.sim_year_select = QComboBox()
        self.sim_year_select.addItem("Simulate Future Year", "future")
        current_year = datetime.now().year
        for i in range(20):
            year = current_year - 1 - i
            self.sim_year_select.addItem(str(year), year)
        layout.addWidget(self.sim_year_select)
        
        self.run_sim_button = QPushButton("Run Full Simulation")
        self.analyze_button = QPushButton("Analyze Last Run")
        self.analyze_button.hide()
        layout.addWidget(self.run_sim_button)
        layout.addWidget(self.analyze_button)
        self.sim_status_label = QLabel("")
        layout.addWidget(self.sim_status_label)
        return panel

    def _create_optimizer_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Strategy Optimizer:"))
        self.optimize_button = QPushButton("Start Self-Improvement")
        self.optimizer_status_label = QLabel("Ready to optimize.")
        self.optimizer_results = QTextEdit()
        self.optimizer_results.setReadOnly(True)
        self.optimizer_results.hide()
        layout.addWidget(self.optimize_button)
        layout.addWidget(self.optimizer_status_label)
        layout.addWidget(self.optimizer_results)
        return panel
        
    def _create_portfolio_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.portfolio_table = QTableWidget()
        self.portfolio_table.setColumnCount(4)
        self.portfolio_table.setHorizontalHeaderLabels(["Symbol", "Qty", "Price", "Value"])
        layout.addWidget(self.portfolio_table)
        return panel

    def _create_news_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.news_status_label = QLabel("Connecting to news stream...")
        self.news_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.news_status_label.setStyleSheet("color: #a0a0b0; margin-bottom: 5px;")
        layout.addWidget(self.news_status_label)
        self.news_feed = QTextEdit()
        self.news_feed.setReadOnly(True)
        layout.addWidget(self.news_feed)
        return panel
        
    def _create_log_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        log_textbox = QTextEditLogger(self)
        log_textbox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_textbox)
        logging.getLogger().setLevel(logging.INFO)
        layout.addWidget(log_textbox.widget)
        return panel
        
    def _create_analysis_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.analysis_content = QWidget()
        self.analysis_layout = QVBoxLayout(self.analysis_content)
        self.analysis_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll_area.setWidget(self.analysis_content)
        layout.addWidget(scroll_area)
        return panel
        
    def _create_chart_panel(self):
        self.chart = QChart()
        self.chart.setBackgroundBrush(QColor("#2a2a44"))
        self.chart.setTitleBrush(QColor("#e0e0e0"))
        self.chart.setTitle("Equity Over Time")
        
        self.series = QLineSeries()
        self.series.setColor(QColor("#4ade80"))
        self.chart.addSeries(self.series)
        
        self.axis_x = QValueAxis()
        self.axis_y = QValueAxis()
        self.axis_x.setLabelFormat("%i")
        self.axis_y.setLabelFormat("$%.0f")
        pen = QColor("#a0a0b0")
        self.axis_x.setLabelsColor(pen)
        self.axis_y.setLabelsColor(pen)
        
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(self.axis_x)
        self.series.attachAxis(self.axis_y)

        chart_view = QChartView(self.chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        return chart_view

    def update_account_display(self, data):
        self.equity_label.setText(f"Equity: ${data['equity']:,.2f}")
        self.cash_label.setText(f"Cash: ${data['cash']:,.2f}")
        self.value_label.setText(f"Value: ${data['portfolio_value']:,.2f}")
        self.date_label.setText(f"Date: {data['date']}")
        
        if data.get('is_sim_update', False) and data.get('equity') is not None:
            if self.series.count() == 0:
                self.series.append(0, 100000)
            self.series.append(self.series.count(), data['equity'])
            if self.series.count() > 252: self.series.remove(0)
            
            # --- THIS IS THE FIX ---
            # Use the correct methods to update chart axes
            axes = self.chart.axes()
            if axes:
                axes[0].setRange(0, self.series.count())
                all_points = [self.series.at(i).y() for i in range(self.series.count())]
                if all_points:
                    axes[1].setRange(min(all_points) * 0.98, max(all_points) * 1.02)
        
    def update_portfolio_display(self, data):
        self.portfolio_table.setRowCount(len(data))
        for row, item in enumerate(data):
            self.portfolio_table.setItem(row, 0, QTableWidgetItem(item['symbol']))
            self.portfolio_table.setItem(row, 1, QTableWidgetItem(f"{item['qty']:.2f}"))
            self.portfolio_table.setItem(row, 2, QTableWidgetItem(f"${item['price']:.2f}"))
            self.portfolio_table.setItem(row, 3, QTableWidgetItem(f"${item['market_value']:.2f}"))

    def update_strategy_status(self, data):
        if data['active']:
            self.strategy_status_label.setText(f"Running: {data['strategy']} ({data['mode']})")
            self.strategy_status_label.setStyleSheet("color: #4ade80;")
        else:
            self.strategy_status_label.setText("Not Running")
            self.strategy_status_label.setStyleSheet("color: #e0e0e0;")

    def update_news_feed(self, data):
        if "Connecting" in self.news_feed.toPlainText():
            self.news_feed.clear()
        self.news_feed.append(f"<b>{data['timestamp']} [{data['sentiment'].upper()}] for {', '.join(data['symbols'])}</b><br>{data['headline']}<br>")

    def update_optimizer_status(self, data):
        self.optimizer_status_label.setText(data['message'])
        if data['status'] == 'complete':
            self.optimizer_results.setText(json.dumps(data['best_params'], indent=2))
            self.optimizer_results.show()
            self.optimize_button.setEnabled(True)

    def update_simulation_status(self, data):
        self.sim_status_label.setText(data['message'])
        if data['status'] == 'complete':
            self.run_sim_button.setEnabled(True)
            self.analyze_button.show()
            
    def update_analysis_display(self, data):
        for i in reversed(range(self.analysis_layout.count())): 
            widget = self.analysis_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        if data.get('error'):
            self.analysis_layout.addWidget(QLabel(f"Error: {data['error']}"))
            return

        best_label = QLabel("Top 3 Best Days")
        best_label.setStyleSheet("font-size: 18px; color: #4ade80; font-weight: bold;")
        self.analysis_layout.addWidget(best_label)
        for day in data.get('best_days', []): self.analysis_layout.addWidget(self._create_day_analysis_widget(day))

        worst_label = QLabel("Top 3 Worst Days")
        worst_label.setStyleSheet("font-size: 18px; color: #f87171; font-weight: bold;")
        self.analysis_layout.addWidget(worst_label)
        for day in data.get('worst_days', []): self.analysis_layout.addWidget(self._create_day_analysis_widget(day))

        rules_label = QLabel("Learned Rules")
        rules_label.setStyleSheet("font-size: 18px; color: #facc15; font-weight: bold;")
        self.analysis_layout.addWidget(rules_label)
        if data.get('learned_rules'):
            for rule in data['learned_rules']:
                rule_text = f"IF news contains '{rule['trigger_keyword']}', THEN {rule['action'].replace('_', ' ')} for {rule['duration_minutes']} mins."
                self.analysis_layout.addWidget(QLabel(rule_text))
        else:
            self.analysis_layout.addWidget(QLabel("No high-confidence rules were generated."))

    def _create_day_analysis_widget(self, day_data):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(f"<b>{day_data['date']}: {day_data['profit_loss']}</b>"))
        layout.addWidget(QLabel(f"<i>Market Context: {day_data['vix_context']}</i>"))

        if day_data['trades']:
            layout.addWidget(QLabel("<b>Trades:</b>"))
            for trade in day_data['trades']: layout.addWidget(QLabel(f"&nbsp;&nbsp;&nbsp;{trade['Side']} {trade['Qty']} {trade['Symbol']} @ ${trade['Price']}"))
        
        if day_data.get('news', {}).get('specific'):
            layout.addWidget(QLabel("<b>Relevant News:</b>"))
            for news in day_data['news']['specific']: layout.addWidget(QLabel(f"&nbsp;&nbsp;&nbsp;[{news['sentiment'].upper()}]: {news['headline']}"))
        
        if day_data.get('news', {}).get('market'):
            layout.addWidget(QLabel("<b>Market News:</b>"))
            for news in day_data['news']['market']: layout.addWidget(QLabel(f"&nbsp;&nbsp;&nbsp;[{news['sentiment'].upper()}]: {news['headline']}"))
        
        return widget

    def start_strategy(self, is_simulation):
        self.stop_strategy()
        preset_name = self.preset_select.currentData()
        strategy_params = STRATEGY_PRESETS[preset_name]["params"]
        strategy_params['symbols'] = SYMBOLS
        
        self.is_simulation = is_simulation
        if self.is_simulation:
            logging.info(f"Starting SIMULATION with '{preset_name}' preset.")
            sim_client = SimulatedAlpacaClient(symbols=SYMBOLS)
            self.active_strategy = OptimizedRenkoStrategy(api_key=None, api_secret=None, base_url=None, client=sim_client, **strategy_params)
        else:
            logging.info(f"Starting LIVE mode with '{preset_name}' preset.")
            self.strategy_running.set()
            self.active_strategy = OptimizedRenkoStrategy(api_key=API_KEY, api_secret=API_SECRET, base_url=BASE_URL, running_event=self.strategy_running, **strategy_params)
            self.strategy_thread = threading.Thread(target=self.run_strategy_async_wrapper, daemon=True)
            self.strategy_thread.start()
            
    def stop_strategy(self):
        self.strategy_running.clear()
        if self.strategy_thread and self.strategy_thread.is_alive():
            self.strategy_thread.join(timeout=2)
        self.active_strategy = None
        self.is_simulation = False
        logging.info("Strategy stopped.")
        self.signals.strategy_update.emit({'active': False})

    def run_full_simulation(self):
        self.start_strategy(is_simulation=True)
        self.series.clear()
        self.series.append(0, 100000)

        year = self.sim_year_select.currentData()
        start_date, end_date = None, None
        if str(year) != "future":
            start_date, end_date = f"{year}-01-01", f"{year}-12-31"
        
        self.sim_status_label.setText(f"Loading data for {year}...")
        self.run_sim_button.setEnabled(False)
        self.analyze_button.hide()
        
        sim_thread = threading.Thread(target=self._run_full_simulation_thread, args=(year, start_date, end_date), daemon=True)
        sim_thread.start()

    def start_optimization(self):
        self.stop_strategy()
        self.optimizer_status_label.setText("Optimization process starting...")
        self.optimize_button.setEnabled(False)
        self.optimizer_results.hide()
        opt_thread = threading.Thread(target=self._run_optimization_thread, daemon=True)
        opt_thread.start()

    def analyze_simulation(self):
        self.analyze_button.setText("Analyzing...")
        self.analyze_button.setEnabled(False)
        analysis_thread = threading.Thread(target=self._run_analysis_thread, daemon=True)
        analysis_thread.start()

    def _run_analysis_thread(self):
        try:
            analyzer = PerformanceAnalyzer('simulation_equity_log.csv', 'simulation_trades.csv', 'simulation_news_log.json', 'simulation_vix_log.csv')
            results = analyzer.run_analysis()
            self.signals.analysis_result.emit(results)
        except Exception as e:
            logging.error(f"Failed to run analysis: {e}")
            self.signals.analysis_result.emit({'error': str(e)})
        finally:
            self.analyze_button.setEnabled(True)
            self.analyze_button.setText("Analyze Last Run")

    async def news_handler(self, news):
        try:
            sentiment = self.news_processor.analyze_sentiment(news.headline)
            self.signals.news_update.emit({'headline': news.headline, 'symbols': news.symbols, 'sentiment': sentiment, 'timestamp': datetime.now().strftime('%H:%M:%S')})
            if self.active_strategy and not self.is_simulation:
                for symbol in news.symbols:
                    if symbol in self.active_strategy.symbols:
                        await self.active_strategy.update_news_sentiment(symbol, sentiment, news.headline)
        except Exception as e:
            logging.error(f"Error processing news: {e}")

    def _start_news_stream(self):
        try:
            logging.info("Starting live news stream...")
            self.news_stream = Stream(API_KEY, API_SECRET, data_feed='sip')
            self.news_stream.subscribe_news(self.news_handler, '*')
            news_thread = threading.Thread(target=self.news_stream.run, daemon=True)
            news_thread.start()
            self.news_status_label.setText("Stream Connected & Listening...")
            self.news_status_label.setStyleSheet("color: #4ade80;")
        except Exception as e:
            logging.error(f"Failed to start news stream: {e}")
            self.news_feed.append("Failed to connect to news stream.")
            self.news_status_label.setText("Stream Connection Failed")
            self.news_status_label.setStyleSheet("color: #f87171;")

    def run_strategy_async_wrapper(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if self.active_strategy and self.strategy_running.is_set():
            loop.run_until_complete(self.active_strategy.run_strategy())
        logging.info("Async strategy loop has exited.")

    async def run_tick_logic(self):
        if not self.active_strategy: return
        client = self.active_strategy.client
        snapshots = client.get_snapshots(self.active_strategy.symbols)
        indicator_bars_data = {symbol: client.get_bars(symbol, TimeFrame.Day, limit=self.active_strategy.adx_period * 3).df for symbol in self.active_strategy.symbols if symbol in client.historical_data}
        daily_bars_data = {symbol: client.get_bars(symbol, TimeFrame.Day, limit=self.active_strategy.long_sma_period + 5).df for symbol in self.active_strategy.symbols if symbol in client.historical_data}
        tasks = []
        for symbol, snapshot in snapshots.items():
            if snapshot and snapshot.minute_bar:
                ohlc_data = {'open': snapshot.minute_bar.o, 'high': snapshot.minute_bar.h, 'low': snapshot.minute_bar.l, 'close': snapshot.minute_bar.c}
                try:
                    position = client.get_position(symbol)
                except (APIError, HTTPError):
                    position = None
                tasks.append(self.active_strategy.run_for_symbol(symbol, ohlc_data, position, indicator_bars_data.get(symbol, pd.DataFrame()), daily_bars_data.get(symbol, pd.DataFrame())))
        if tasks: await asyncio.gather(*tasks)

    def _run_full_simulation_thread(self, year, start, end):
        client = self.active_strategy.client
        if str(year) == "future":
            if not client.generate_future_data():
                self.signals.simulation_status.emit({'status': 'error', 'message': 'Failed to generate future data.'})
                return
        else:
            if not client.load_historical_data(start=start, end=end):
                self.signals.simulation_status.emit({'status': 'error', 'message': 'Failed to load data.'})
                return
        
        while client.tick() and self.is_simulation:
            asyncio.run(self.run_tick_logic())
            if client.current_tick % 5 == 0:
                self.signals.account_update.emit(self.get_account_data(is_sim=True))
        
        self.signals.account_update.emit(self.get_account_data(is_sim=True))
        self.signals.simulation_status.emit({'status': 'complete', 'message': f'Finished simulation for {year}.'})
        logging.info("Full simulation run has completed.")

    def _run_optimization_thread(self):
        logging.info("Starting optimization engine...")
        optimizer = OptimizationEngine(SYMBOLS, self.signals)
        asyncio.run(optimizer.run_optimization())
        
    def get_account_data(self, is_sim=False):
        if self.active_strategy:
            account = self.active_strategy.get_account_info()
            if account:
                data = {'equity': float(account.equity), 'cash': float(account.cash), 'portfolio_value': float(account.portfolio_value), 'is_sim_update': is_sim}
                if self.is_simulation and hasattr(self.active_strategy.client, 'current_date') and self.active_strategy.client.current_date:
                    data['date'] = self.active_strategy.client.current_date.strftime('%Y-%m-%d')
                else:
                    data['date'] = 'Live'
                return data
        return {'equity': 0, 'cash': 0, 'portfolio_value': 0, 'date': 'N/A', 'is_sim_update': is_sim}

    def closeEvent(self, event):
        self.stop_strategy()
        if self.news_stream:
            self.news_stream.stop()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TradingBotGUI()
    window.show()
    sys.exit(app.exec())
