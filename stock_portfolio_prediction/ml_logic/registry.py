import glob
import os
import time
import pickle

#from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage

BUCKET_NAME = 'stock_portfolio_prediction'
LOCAL_REGISTRY_PATH = '/Users/kychee2013/Desktop/Stock_Portfolio_Prediction/Model'

def save_model(stock):
    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{stock.ticker}.h5")
    stock.model.save(model_path)

    model_filename = f"{stock.ticker}.h5" # e.g. "20230208-161047.h5" for instance
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    print(client.get_bucket(BUCKET_NAME))
    blob = bucket.blob(f"models/{model_filename}")
    print(blob)
    blob.upload_from_filename(model_path)

    print("✅ Model saved to GCS")

    return None

def load_model(stock):

    try:
        model_path = f'gs://{BUCKET_NAME}/models/{stock.ticker}.h5'
        model = keras.models.load_model(model_path)
        print("✅ Model loaded from cloud storage")
        return model

    except:
        print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")
        return None
