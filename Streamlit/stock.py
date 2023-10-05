import numpy as np
import pandas as pd


import yfinance as yf
import talib


class Stock:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = pd.DataFrame() # All data related to the stock (historical data + technical indicators)
        self.tensor = np.array([])

        self.technical_indicators = pd.DataFrame() # All technical indicators removing all NaN value
        self.today_technical_indicators = pd.Series() # All techincal indicators for today
        self.price = 0.0
        self.labels = pd.Series() # Increased or decreased, compare to the stock price from 10 days ago
        self.prediction = 0.0
        self.model = {}
        self.pred = pd.DataFrame()


    def get_historical_data(self, period):
        self.data = yf.download(self.ticker, period=period, progress=False)
        print(self.data)

    def get_technical_indicators(self):
        self.data['SMA_20'] = talib.SMA(self.data['Close'], timeperiod=20)
        self.data['SMA_50'] = talib.SMA(self.data['Close'], timeperiod=50)
        self.data['SMA_200'] = talib.SMA(self.data['Close'], timeperiod=200)
        self.data["RSI"] = talib.RSI(self.data["Close"], timeperiod=14)
        self.data["upper_band"], self.data["middle_band"], self.data["lower_band"] = talib.BBANDS(self.data["Close"], timeperiod=20)
        self.data["macd"], self.data["macd_signal"], self.data["macd_hist"] = talib.MACD(self.data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
        self.data["stochastic_k"], self.data["stochastic_d"] = talib.STOCH(self.data["High"], self.data["Low"], self.data["Close"], fastk_period=14, slowk_period=3, slowd_period=3)

        train_data_aux = self.data[['Close', "SMA_20", "SMA_50", "SMA_200", "upper_band", "middle_band", "lower_band",
                                "RSI", "macd", "macd_signal", "macd_hist", "stochastic_k", "stochastic_d"]].dropna()

        self.technical_indicators = train_data_aux.iloc[:-10, 1:]

        labels_aux = (train_data_aux['Close'].shift(-10) > train_data_aux['Close']).astype(int)
        self.labels = labels_aux[:-10]

        self.today_technical_indicators = self.data[["SMA_20", "SMA_50", "SMA_200", "upper_band", "middle_band",
                                                   "lower_band", "RSI", "macd", "macd_signal", "macd_hist",
                                                   "stochastic_k", "stochastic_d"]].iloc[-1]
        self.price = self.data["Close"].iloc[-1]
