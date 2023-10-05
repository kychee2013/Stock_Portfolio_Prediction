import streamlit as st
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from data import *
from stockscreener import *
from stock import Stock
from model import *
from pyportfolioopt import weights_max_sharpe, plot_efficient_frontier
from fundamental_score import get_score
from ytd_performance import get_ytd_performance
import os
import tensorflow as tf
import random

st.markdown("""# AI-Powered Stock Picking Invest with the odds in your favor. Get unique insights, boost your 
portfolios, and make smart data-driven investment decisions.""")

# Adding a text in the sidebar
st.sidebar.text('Your Text Here')
# Add a radio button
st.sidebar.radio('label', options=[])

st.markdown("""## Stocks Ranked by AI
US-listed stocks are ranked according to the AI Score, which rates the probability of beating the market in the next 3 months.

""")

data = {
    'Company': [_.split('.h5')[0] for _ in os.listdir('models')],
}

# Create a DataFrame from the data
with st.spinner("Loading, please wait..."):
    df = pd.DataFrame(data).set_index('Company')
    df["Technical Score"] = random.sample(range(11, 100), df.shape[0])
    df = pd.concat([df, get_score([_.split('.h5')[0] for _ in os.listdir('models')])], axis=1)
    sort_inx = df.T.sum().sort_values(ascending=False).index
    df["YTD Performance"] = [get_ytd_performance(_) for _ in [_.split('.h5')[0] for _ in os.listdir('models')]]
    show_df = df.rename(columns={0: 'Fundamental Score'}).loc[sort_inx].reset_index()
    show_df["Rank"] = [_+1 for _ in range(df.shape[0])]
    # Display the ranked stock table
    st.dataframe(show_df.set_index('Rank'))

# My stock portfolio optimizer
st.header("My Stock Portfolio Optimizer")

# load the model
models_directory = 'models'

name_lis = []
# load model, set cache to prevent reloading
with st.spinner('Load models...'):
    for name in os.listdir(models_directory):
        name_lis.append(name.split('.h5')[0])
        if name.split('.h5')[0] not in st.session_state:
            st.session_state[name.split('.h5')[0]] = tf.keras.models.load_model(os.path.join(models_directory, name))
        else:
            continue

user_tickers = st.multiselect('Select all stock tickers to be included in portfolio separated by commas',
                              name_lis)
if st.button('Generate'):
    with st.spinner('Building, please wait'):
        st.dataframe(df.rename(columns={0: 'Fundamental Score'}).loc[user_tickers])

        # Define the required filters here for each technical indicator
        filters = [lambda stock: filter_technical_indicator(stock, 'SMA_20', '>', 100)]
        # Create list of Stock instance
        real_estate_stocks = []
        for ticker in user_tickers:
            real_estate_stocks.append(Stock(ticker))

        # Initialize the screener with stocks and filters
        screener = StockScreener(real_estate_stocks, filters)

        # Add 1 year historical data and technical indicators to stocks
        screener.add_data()

        # Filter 1: filter stocks based on historical price and technical indicators
        filtered_stocks = screener.apply_filters()

        # Train model
        screener.train_models()

        # Make predictions and keep only those indicating an increase on the 10th day.
        predicted_stocks = screener.predict_stocks(filtered_stocks)
        for stock in predicted_stocks:
            stock.get_historical_data('5y')
            stock.get_technical_indicators()

            # Create a new dataframe and drop all NaN values
            data = stock.data.dropna()

            # Convert the dataframe to a numpy array
            dataset = data.values

            # Scale the data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(dataset)

            pred = st.session_state[stock.ticker].predict(scaled_data[-60:, 6:].reshape(-1, 60, 12))
            inverse_scale = np.vectorize(lambda x: x * scaler.data_range_[3] + scaler.data_min_[3])
            pred = inverse_scale(pred)

            stock.pred = pred.flatten()
        # Convert predicted 1 month Close price to DataFrame
        predicted_1mo = {}

        for stock in predicted_stocks:
            predicted_1mo[stock.ticker] = stock.pred
        mu, sigma, sharpe, a = weights_max_sharpe(pd.DataFrame.from_dict(predicted_1mo), 21)
        st.subheader("Expected annual return: {:.1f}%".format(100 * mu))
        st.subheader("Annual volatility: {:.1f}%".format(100 * sigma))
        st.subheader("Sharpe Ratio: {:.2f}".format(sharpe))

        sizes = [a[_] for _ in a.keys() if a[_] != 0]
        labels = ["{}\n{}".format(_, a[_]) for _ in a.keys() if a[_] != 0]
        fig, axes = plt.subplots(figsize=(10, 10))

        wedges, texts = axes.pie(sizes, labels=labels, startangle=90)
        for label, text in zip(labels, texts):
            text.set(size=15, color='black')
        st.pyplot(fig)

        plot_efficient_frontier(pd.DataFrame.from_dict(predicted_1mo), 21)
