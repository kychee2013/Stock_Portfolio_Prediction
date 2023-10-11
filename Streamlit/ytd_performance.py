import datetime
import yfinance as yf
import numpy as np


def get_ytd_performance(stock_symbol):
    # Define the start date for YTD calculation
    start_date = datetime.datetime(datetime.datetime.now().year, 1, 1)  # YTD start date

    try:
        # Fetch historical stock data from Yahoo Finance
        stock_data = yf.download(stock_symbol, start=start_date)

        # Calculate YTD performance
        ytd_start_price = stock_data['Adj Close'][0]
        ytd_end_price = stock_data['Adj Close'][-1]
        ytd_performance = (ytd_end_price - ytd_start_price) / ytd_start_price * 100

        return ytd_performance

    except Exception as e:
        return np.NaN
