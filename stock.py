import pandas as pd

import yfinance as yf
import talib


class Stock:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = pd.DataFrame()

        # Deep Learning Attributes
        self.technical_indicators = pd.DataFrame()
        self.labels = pd.DataFrame()
        self.prediction = 0.0


    def get_historical_data(self):
        self.data = yf.download(self.ticker, period='5y')

    def get_technical_indicators(self):
        self.data['SMA_20'] = talib.SMA(self.data['Close'], timeperiod=20)
        self.data['SMA_50'] = talib.SMA(self.data['Close'], timeperiod=50)
        self.data['SMA_200'] = talib.SMA(self.data['Close'], timeperiod=200)
        self.data["RSI"] = talib.RSI(self.data["Close"], timeperiod=14)
        self.data["upper_band"], self.data["middle_band"], self.data["lower_band"] = talib.BBANDS(self.data["Close"], timeperiod=20)
        self.data["macd"], self.data["macd_signal"], self.data["macd_hist"] = talib.MACD(self.data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
        self.data["stochastic_k"], self.data["stochastic_d"] = talib.STOCH(self.data["High"], self.data["Low"], self.data["Close"], fastk_period=14, slowk_period=3, slowd_period=3)
