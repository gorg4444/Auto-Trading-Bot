import logging

class BaseStrategy:
    """A simple base class for trading strategies."""
    def __init__(self, api_key, api_secret, base_url, symbols, debug=False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.symbols = symbols
        self.debug = debug
        self.client = None