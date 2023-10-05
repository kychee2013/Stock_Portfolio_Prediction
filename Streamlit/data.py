import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(stock):
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
    stocks = pd.read_csv('Data/stock_all.csv', index_col=0)
    stocks = stocks.dropna(subset=['Sector']).reset_index(drop=True)
    mapping_dict = {'Industrials': 'manufacturing',
                    'Real Estate': 'real_estate',
                    'Miscellaneous': 'NaN',
                    'Consumer Discretionary': 'retail_wholesale',
                    'Finance': 'finance',
                    'Health Care': 'life_sciences',
                    'Consumer Staples': 'retail_wholesale',
                    'Utilities': 'energy_transportation',
                    'Technology': 'technology',
                    'Energy': 'energy_transportation',
                    'Telecommunications': 'technology',
                    'Basic Materials': 'manufacturing'}
    stocks['Topics'] = stocks['Sector'].map(mapping_dict)
    stocks = stocks.drop(stocks[stocks['Topics'] == 'NaN'].index).reset_index(drop=True)
    return stocks

def load_data():
    """
    load data to database
    """
    pass
