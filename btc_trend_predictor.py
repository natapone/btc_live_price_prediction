import pandas as pd
import numpy as np
import joblib
import requests
import json
import datetime
import talib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from tensorflow.keras.models import load_model

class BtcTrendPredictor:
    def __init__(self, model_name = 'tmp_model', model_path = './models/'):
        # Load the model
        # self.model = load_model(model_path)

        # # Load the scaler and encoder
        # self.scaler = joblib.load("./models/tmp_model_scaler.joblib")
        # self.encoder = joblib.load("./models/tmp_model_encoder.joblib")

        self.T = 72 # number of periods to use as input features (time steps)
        self.predict_period = 1 # number of periods to predict
        self.C = 6 # number of classes

    def remove_outlier(self, df, iqr_threshold = 5):
    
        # Calculate the first quartile (25th percentile) and third quartile (75th percentile)
        q1 = df['volume'].quantile(0.25)
        q3 = df['volume'].quantile(0.75)

        # Calculate the interquartile range (IQR)
        iqr = q3 - q1

        # Define lower and upper bounds for outliers
        lower_bound = q1 - iqr_threshold * iqr
        upper_bound = q3 + iqr_threshold * iqr

        lower_bound = 0 if lower_bound < 0 else lower_bound

        # remove outliers from df
        df = df[(df['volume'] > lower_bound) & (df['volume'] < upper_bound)]
        
        return df

    def read_hist_data(self, name = 'BTC', timeframe = '1h'):
        file_path = f"./data/{name}_USDT-{timeframe}.json"
        df = pd.read_json(file_path)

        # set column names
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

        # convert unix timestamp to datetime
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')

        # change datetime to index
        df.set_index('datetime', inplace=True)
        
        df = self.remove_outlier(df)
        return df

    def get_target(self, df, target_shift = 3):
    
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
        df['target'] = np.select(conditions, values, default=1,)

        # shift target
        df['target'] = df['target'].shift(target_shift * -1)

        # drop columns
        df.drop(columns=['ohlc', 'ema_10', 'ema_20', 'ema_diff'], inplace=True)

        return df
    
    def get_features_v1(self, df, T=3):

        df = df.copy()

        # ++ Features ++
        inputs = ['open', 'high', 'low', 'close', 'volume'] 

        # log volume
        df.loc[:, 'volume'] = np.log(df['volume'])

        # range from 1 to T
        lags = [x for x in range(0, T)]

        # loop lags
        for lag in lags:
            # loop periods and inputs
            for input in inputs:
                if lag == 0:
                    # % of change
                    column_name = f'{input}_lag{lag}'
                    df.insert(0, column_name, df[input].pct_change(periods=1))

                else:
                    # use lagged price, sort from oldest to newest
                    column_name = f'{input}_lag{lag}'
                    df.insert(0, column_name, df[f'{input}_lag0'].shift(lag))

            df = df.copy()

        # drop unused columns
        drop_columns = ['open', 'high', 'low', 'close','volume']
        df.drop(columns=drop_columns, inplace=True)

        return df
    
    def get_training_data(self):
        df = self.read_hist_data()
        df = self.get_target(df, target_shift = self.predict_period)
        df = self.get_features_v1(df, T=self.T)
        df = df.dropna()

        return df

    def train_dl_trend_prediction_model(self, df, T, model_name = 'tmp_model', feature_function='get_features_v1', test_size=0.2, epochs=300):
        # 1. Get features and target
        # Get features
        X = df.copy()
        X = globals()[feature_function](X, T=T)
        X.dropna(inplace=True)

        # Get target
        y = df.copy()
        y = self.get_target(y, target_shift = self.predict_period)
        y.dropna(inplace=True)

        # 2. Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        # 3. Scale the data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 4. Reshape the data
        X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], T, 1)
        X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], T, 1)

        # 5. One-hot encode the target
        encoder = OneHotEncoder(sparse=False)
        y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
        y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1))

        # 6. Build the model
        model = Sequential()
        model.add(LSTM(50, input_shape=(T, 1)))
        model.add(Dense(self.C, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # 7. Fit the model
        model.fit(X_train_scaled, y_train_encoded, epochs=epochs, validation_data=(X_test_scaled, y_test_encoded), verbose=1)

        # 8. Save the model, scaler and encoder
        model.save(f"./models/{model_name}.h5")
        joblib.dump(scaler, f"./models/{model_name}_scaler.joblib")
        joblib.dump(encoder, f"./models/{model_name}_encoder.joblib")

        return model

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

    # def get_target(self, df, target_shift = 2):
    #     # def get_target_next_ema_diff_v2(df, target_shift = 3):
        
    #     target_threshold = 0.2
    #     # oclh
    #     df['ohlc'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    #     # diff between ema 10 and 20
    #     df['ema_10'] = talib.EMA(df['ohlc'], timeperiod=10)
    #     df['ema_20'] = talib.EMA(df['ohlc'], timeperiod=20)
    #     df['ema_diff'] = (df['ema_10'] - df['ema_20']) / df['ohlc'] * 100

    #     conditions = [
    #         (df['ema_diff'].isnull()),
    #         (df['ema_diff'] > target_threshold) & (df['ema_diff'] > df['ema_diff'].shift(1)),
    #         (df['ema_diff'] > target_threshold) & (df['ema_diff'] <= df['ema_diff'].shift(1)),
    #         (df['ema_diff'] > 0) & (df['ema_diff'] <= target_threshold),
    #         (df['ema_diff'] < target_threshold * -1) & (df['ema_diff'] < df['ema_diff'].shift(1)),
    #         (df['ema_diff'] < target_threshold * -1) & (df['ema_diff'] >= df['ema_diff'].shift(1)),
    #         (df['ema_diff'] <= 0) & (df['ema_diff'] >= target_threshold * -1)
    #     ]
    #     values = [np.nan, 3, 2, 1, -3, -2, -1]
    #     df['target'] = np.select(conditions, values, default=0,)

    #     # shift target
    #     df['target'] = df['target'].shift(target_shift * -1)

    #     # drop columns
    #     df.drop(columns=['ohlc', 'ema_10', 'ema_20', 'ema_diff'], inplace=True)

    #     return df

    # def preprocess_data(self, df):
    #     df = get_target(df)  # Use the target function you provided
    #     df = get_features(df, self.T)  # Use the features function you provided
    #     df.drop(columns=['target'], inplace=True, errors='ignore')  # Drop the target column
    #     return df

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

# === Usage ===
predictor = BtcTrendPredictor()


# == Train ==
# df = predictor.read_hist_data()
# df = predictor.get_target(df)
# df = predictor.get_features_v1(df, T=3)

df = predictor.get_training_data()
print(df.head(10))
print(df.tail(10))

# model =  predictor.train_dl_trend_prediction_model(df, T, model_name = 'dl_btc_ema_2hr_trend', test_size=0.2, epochs=300)


# == Predict ==

# df = predictor.fetch_live_binance_data()
# print(df)

# df = predictor.predict()
# print(df)

# df = predictor.get_target(df)
# print(df)