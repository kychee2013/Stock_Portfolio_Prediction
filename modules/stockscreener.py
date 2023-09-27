import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def filter_sector(stock, sector):
    return stock.sector == sector

def filter_price(stock, min_price, max_price):
    return min_price <= stock.price <= max_price

def filter_technical_indicator(self, indicator_name, operator, value):
    if indicator_name not in self.today_technical_indicators:
        print(f"{indicator_name} not found for {self.ticker}")
        return False

    # Obtain the value of the technical indicator
    indicator_value = self.today_technical_indicators[indicator_name]

    # Compare according to operator
    if operator == '>':
        return float(indicator_value) > value
    elif operator == '>=':
        return float(indicator_value) >= value
    elif operator == '<':
        return float(indicator_value) < value
    elif operator == '<=':
        return float(indicator_value) <= value
    elif operator == '==':
        return float(indicator_value) == value
    else:
        return False

class StockScreener:
    def __init__(self, stocks, filters):
        self.stocks = stocks
        self.filters = filters
        self.models = {}

    def add_data(self):
        for stock in self.stocks:
            stock.get_historical_data('1y')
            stock.get_technical_indicators()

    def apply_filters(self):
        filtered_stocks = []
        for stock in self.stocks:
            passed_all_filters = True
            for filter_func in self.filters:
                if not filter_func(stock):
                    passed_all_filters = False
                    print(f"{stock.ticker} failed to pass filter")
                    break
            if passed_all_filters:
                filtered_stocks.append(stock)
        return filtered_stocks
