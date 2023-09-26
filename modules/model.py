from typing import Tuple

import numpy as np
from tensorflow import keras
from keras import Model, Sequential, layers, optimizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

def initialize_model(input_shape: tuple):
    """
    Initialize the Neural Network with random weights
    """
    model = Sequential()
    model.add(layers.LSTM(units=100, return_sequences=True, activation='tanh', input_shape=input_shape))
    model.add(layers.LSTM(units=50, return_sequences=False))
    model.add(layers.Dense(25, activation='linear'))
    model.add(layers.Dense(1, activation='linear'))

    print("✅ Model initialized")

    return model

def compile_model(model: Model, learning_rate=0.001):
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        epochs=100,
        patience=2,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        validation_split=validation_split,
        callbacks=[es],
        verbose=0
    )

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history

def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        scaler: MinMaxScaler,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the test data
    """

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    predictions = model.predict(X) # --> has shape (n,1)

    # To use scaler.inverse_transform(), need to add one empty column in front to make the shape as (n,2)
    pred = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0],1)))))[:, 0]
    real = scaler.inverse_transform(np.hstack((y.reshape(-1,1), np.zeros((predictions.shape[0],1)))))[:, 0]

    # Calculate the root mean square error (rmse)
    rmse = np.sqrt(np.mean((pred - real)**2))
    print (f'✅ Model evaluated, RMSE: {round(rmse, 2)}')

    return pred, rmse
