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
        self.filtered_stocks = []
        self.models = {}

    def add_data(self):
        for stock in self.stocks:
            stock.get_historical_data('1y')
            stock.get_technical_indicators()

    def apply_filters(self):
        for stock in self.stocks:
            passed_all_filters = True
            for filter_func in self.filters:
                if not filter_func(stock):
                    passed_all_filters = False
                    # print(f"{stock.ticker} failed to pass filter")
                    break
            if passed_all_filters:
                self.filtered_stocks.append(stock)
        return self.filtered_stocks

    def create_model(self, train_data):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_shape=(train_data.shape[1],), activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_models(self):
        # for stock in self.filtered_stocks:
        for stock in self.stocks:
            train_data = stock.technical_indicators
            train_labels = stock.labels

            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data.values)
            train_labels = np.array(train_labels)

            model = self.create_model(train_data)
            history = model.fit(train_data, train_labels, epochs=10, verbose=0)
            self.models[stock.ticker] = (model, scaler)
        return model, history

    def predict_stocks(self, new_stocks):
        predicted_stocks = []
        for stock in new_stocks:
            if stock.ticker in self.models:
                model, scaler = self.models[stock.ticker]
                new_features_aux = np.array(stock.today_technical_indicators).reshape(1, -1)
                new_stock_data = scaler.transform(new_features_aux)
                prediction = model.predict(new_stock_data)
                stock.prediction = prediction
                if prediction > 0.5:
                    predicted_stocks.append(stock)

        return predicted_stocks
