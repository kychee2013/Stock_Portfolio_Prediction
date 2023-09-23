from stock import Stock

import numpy as np
from sklearn.preprocessing import MinMaxScaler

def clean_data(stock:Stock):
    """
    Input: stock in filtered_stocks list
    """
    sc = MinMaxScaler()
    stock.tensor = sc.fit_transform(stock.data)
    stock.tensor = np.nan_to_num(stock.tensor, nan=-1)

def get_data():
    """
    load data from database
    """
    pass

def load_data():
    """
    load data to database
    """
    pass
