Self-Optimizing Renko Trading Bot

This is an advanced, event-driven trading bot that uses a customizable Renko charting strategy combined with a self-optimizing genetic algorithm and real-time news analysis to make trading decisions. The application is built with a native desktop GUI using PyQt6.

Key Features

-Adaptive Renko Strategy: The core trading logic is based on Renko bricks, but the brick size is not fixed. It dynamically adapts based on the market regime (trending or ranging), which is determined by the ADX indicator.

-Self-Improving Genetic Algorithm: The bot includes a powerful optimization engine that runs hundreds of simulations over 20+ years of historical data to find the most robust and profitable strategy parameters. It uses a Sharpe Ratio fitness score to prioritize risk-adjusted returns.

-Live News Analysis: The bot connects to a live news stream, performs real-time sentiment analysis on headlines, and uses a "two-factor" system to react to significant market events. It also learns defensive rules from historical news correlations.

-Advanced Risk Management: Implements portfolio-level risk controls, including a limit on the maximum number of concurrent positions and total capital allocation, to prevent over-exposure.

-Comprehensive Backtesting Suite: The GUI allows for detailed backtesting on any of the last 20 years, as well as a "Future Year Simulation" that generates a new, statistically plausible price path based on historical volatility and drift.

-Performance Analyzer: After each simulation, a diagnostic engine analyzes the best and worst trading days, correlating them with the trades made, relevant news events, and the market's volatility (VIX) at the time.

Requirements
Python 3.11+

An Alpaca Paper Trading Account for live trading and historical data.

A free API key from Polygon.io for the historical news analysis.

Installation and Startup
This project includes automated setup scripts for both Windows and Linux/macOS.

1. Set Up Your API Keys
Before running, you must create a file named .env in the main project directory. This file stores your secret API keys and should never be shared. Add your keys to it like this:

# Alpaca API Keys
ALPACA_API_KEY="YOUR_ALPACA_KEY_HERE"
ALPACA_SECRET_KEY="YOUR_ALPACA_SECRET_KEY_HERE"

# Polygon.io API Key (for news analysis)
POLYGON_API_KEY="YOUR_POLYGON_KEY_HERE"

2. Run the Automated Script
On Windows:
Simply double-click the install_and_run.bat file.

This script will automatically:

Create a local Python virtual environment (venv).

Install all required libraries from requirements.txt.

Launch the desktop application.

On macOS & Linux:
Open your terminal in the project directory.

Make the script executable by running:

chmod +x install_and_run.sh

Run the script:

./install_and_run.sh

This script performs the same setup steps as the Windows version.
