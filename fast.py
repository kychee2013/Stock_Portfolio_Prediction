import pandas as pd
import numpy as np
from modules.stock import Stock

from modules.registry import load_model
from sklearn.preprocessing import MinMaxScaler

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?ticker=AAPL,GOOG,ALX,AMT,AVB,DLR,EGP,EXR,MAA,SUI
@app.get("/predict")
def predict():
    tickers = locals()['ticker']
    predicted_1mo = {}
    for ticker in tickers.split(','):
        stock = Stock(ticker)

        # Retrieve historical data
        stock.get_historical_data('5y')
        stock.get_technical_indicators()

        # Process data
        data = stock.data.dropna()
        dataset = data.values

        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(dataset)

        # Load model
        model = load_model(stock)
        assert model is not None

        # Predict future 1 month close price
        pred = model.predict(scaled_data[-60:, 6:].reshape(-1, 60, 12))
        inverse_scale = np.vectorize(lambda x: x*scaler.data_range_[3] + scaler.data_min_[3])
        pred = inverse_scale(pred)
        stock.pred = pred.flatten()

        predicted_1mo[stock.ticker] = stock.pred

    return predicted_1mo


@app.get("/")
def root():
    return dict(greeting="Hello")
