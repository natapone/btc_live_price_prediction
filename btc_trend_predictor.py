import pandas as pd
import numpy as np
import joblib
import requests
import json
import datetime
import talib

from tensorflow.keras.models import load_model

class BtcTrendPredictor:
    def __init__(self, model_path="./models/dl_final_model.kera"):
        # Load the model
        # self.model = load_model(model_path)

        # # Load the scaler and encoder
        # self.scaler = joblib.load("./models/tmp_model_scaler.joblib")
        # self.encoder = joblib.load("./models/tmp_model_encoder.joblib")
        self.T = 72

    # def fetch_live_data(self, ):
    #     # Fetch live data from Binance
    #     url = f"https://api.binance.com/api/v3/klines?symbol={BTCUSDT}&interval={1h}"
    #     data = requests.get(url).json()
    #     df = pd.DataFrame(data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    #     df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    #     df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    #     df.set_index('datetime', inplace=True)
    #     return df
    
    def format_live_binance_data(self, df):

        # use only completed hour
        df = df[df['close_time'] <= datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")] 

        df_live = pd.DataFrame()
        df_live['datetime'] = df['open_time']
        df_live['open'] = df['open']
        df_live['high'] = df['high']
        df_live['low'] = df['low']
        df_live['close'] = df['close']
        df_live['volume'] = df['volume']
        
        return df_live

    def fetch_live_binance_data(self, name = 'BTC', timeframe = '1h', limit = 1000):
        url = f"https://api.binance.com/api/v3/klines?symbol={name}USDT&interval={timeframe}&limit={limit}"
        response = requests.get(url)
        df = pd.DataFrame(json.loads(response.text))
        df.columns = ['open_time',
                    'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        
        df = df.astype(float)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        df = self.format_live_binance_data(df)

        # set datetime as index
        df = df.set_index('datetime')


        return df

    def get_target(self, df, target_shift = 2):
        # def get_target_next_ema_diff_v2(df, target_shift = 3):
        
        target_threshold = 0.2
        # oclh
        df['ohlc'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        # diff between ema 10 and 20
        df['ema_10'] = talib.EMA(df['ohlc'], timeperiod=10)
        df['ema_20'] = talib.EMA(df['ohlc'], timeperiod=20)
        df['ema_diff'] = (df['ema_10'] - df['ema_20']) / df['ohlc'] * 100

        conditions = [
            (df['ema_diff'].isnull()),
            (df['ema_diff'] > target_threshold) & (df['ema_diff'] > df['ema_diff'].shift(1)),
            (df['ema_diff'] > target_threshold) & (df['ema_diff'] <= df['ema_diff'].shift(1)),
            (df['ema_diff'] > 0) & (df['ema_diff'] <= target_threshold),
            (df['ema_diff'] < target_threshold * -1) & (df['ema_diff'] < df['ema_diff'].shift(1)),
            (df['ema_diff'] < target_threshold * -1) & (df['ema_diff'] >= df['ema_diff'].shift(1)),
            (df['ema_diff'] <= 0) & (df['ema_diff'] >= target_threshold * -1)
        ]
        values = [np.nan, 3, 2, 1, -3, -2, -1]
        df['target'] = np.select(conditions, values, default=0,)

        # shift target
        df['target'] = df['target'].shift(target_shift * -1)

        # drop columns
        df.drop(columns=['ohlc', 'ema_10', 'ema_20', 'ema_diff'], inplace=True)

        return df

    def preprocess_data(self, df):
        df = get_target(df)  # Use the target function you provided
        df = get_features(df, self.T)  # Use the features function you provided
        df.drop(columns=['target'], inplace=True, errors='ignore')  # Drop the target column
        return df

    def predict(self):
        # 1. Fetch live data
        live_data = self.fetch_live_binance_data()

        # 2. Get features and target
        # Get features
        X_live = df_live.copy()
        X_live = globals()[feature_function](X_live, T=T)
        X_live.dropna(inplace=True)

    #     # 2. Preprocess the live data
    #     preprocessed_data = self.preprocess_data(live_data)

    #     # Ensure you have the last batch of size T for prediction
    #     if preprocessed_data.shape[0] < self.T:
    #         raise ValueError("Not enough live data to make a prediction.")

    #     input_data = preprocessed_data[-self.T:].values
    #     input_data = self.scaler.transform(input_data).reshape(1, self.T, -1)

    #     # 3. Predict
    #     prediction = self.model.predict(input_data)
    #     trend = self.encoder.inverse_transform(prediction)
    #     return trend[0][0]

        return FileNotFoundError

# Usage
predictor = BtcTrendPredictor()
# print(predictor.predict())

df = predictor.fetch_live_binance_data()
print(df)

df = predictor.predict()
print(df)

df = predictor.get_target(df)
print(df)