import pandas as pd
import numpy as np
import joblib
import requests
import json
import datetime
import talib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from tensorflow.keras.models import load_model


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2

import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib

class BtcTrendPredictor:
    def __init__(self, predict_period = 1, model_path = './models/'):

        self.T = 72 # number of periods to use as input features (time steps)
        self.predict_period = predict_period # number of periods to predict
        self.C = 6 # number of classes
        self.model_path = model_path
        self.model_name = f'btc_live_pred_{predict_period}hr'

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
    
    def plot_rnn_training_eval(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
        plt.legend()
        plt.title('Loss Over Epochs')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
        plt.legend()
        plt.title('Accuracy Over Epochs')

        plt.tight_layout()
        plt.show()

        return None

    def eval_classification_model(self, model, X_test, y_test, one_hot_encoder):
        y_pred = model.predict(X_test, verbose=0)

        # reverse one-hot encoding
        y_test = one_hot_encoder.inverse_transform(y_test)
        y_pred = one_hot_encoder.inverse_transform(y_pred)

        fig, ax = plt.subplots(1, 2, figsize=(15, 7))

        classes = ['-3.0', '-2.0', '-1.0', '1.0', '2.0', '3.0']

        # Confusion Matrix
        matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', ax=ax[0])
        ax[0].set_xticklabels(classes, rotation=45)
        ax[0].set_yticklabels(classes, rotation=0)
        ax[0].set_xlabel('Predicted')
        ax[0].set_ylabel('True')
        ax[0].set_title("Confusion Matrix")

        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True, target_names=classes)
        row_data = [["", "precision", "recall", "f1-score", "support"]]
        for k, v in report.items():
            if k not in ["accuracy"]:
                row_data.append([f"{k}"] + [f'{v[i]:.2f}' for i in ["precision", "recall", "f1-score", "support"]])
        row_data.append(["accuracy", "", "", f"{report['accuracy']:.2f}", ""])

        ax[1].axis('off')
        table = ax[1].table(cellText=row_data, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table.scale(1, 2)

        # Style the table with similar colors of the heatmap
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor('#2c7bb6')  # Header cells color
            else:
                cell.set_facecolor('#f5f5f5')  # Rest of the table color
            cell.set_edgecolor('w')  # Border color

        table.auto_set_column_width(col=list(range(5)))
        ax[1].set_title("Classification Report")

        plt.tight_layout()
        plt.show()

        return None

    def train_dl_trend_prediction_model(self, model_path = './models/', test_size=0.1, random_state = 55, epochs=50, show_eval=True, verbose=0):
        
        # , df, T, model_name = 'tmp_model'
        
        time_start = time.time()

        # Training data
        df = self.get_training_data()

        # Pre-processing 
        drop_columns = ['target']
        X = df.drop(columns=drop_columns)

        # One-hot encode labels
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(df['target'].values.reshape(-1, 1))

        # save encoder
        encoder_filepath = model_path + self.model_name + '_encoder.joblib'
        joblib.dump(encoder, encoder_filepath)

        # Dimension parameters
        F = int((df.shape[1] -1) / self.T) # number of input features 
        C = y.shape[1] # number of classes
        # print(f"Input shape: ({self.T}, {F})")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)

        # Fit the scaler using the training data
        scaler = StandardScaler()
        
        X_train_pre = scaler.fit_transform(X_train)
        X_test_pre = scaler.transform(X_test)

        # save scaler
        scaler_filepath = model_path + self.model_name + '_scaler.joblib'
        joblib.dump(scaler, scaler_filepath)

        X_train_pre = X_train_pre.reshape(-1, self.T, F)
        X_test_pre = X_test_pre.reshape(-1, self.T, F)

        # data set = (X_train_pre, y_train), (X_test_pre, y_test)

        # Deep learning model
        input_shape = (self.T, F) # (time_steps, features)


        # Create the LSTM model
        model = Sequential()

        model.add(LSTM(units=60, return_sequences=True, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01), input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(BatchNormalization()) 

        model.add(LSTM(units=30, return_sequences=True, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(BatchNormalization()) 

        model.add(LSTM(units=30, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(BatchNormalization()) 

        # Output layer
        model.add(Dense(units=C, activation='softmax'))

        # Compile the model
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)     # set learning rate
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(X_train_pre, y_train, epochs=epochs, batch_size=128, validation_data=(X_test_pre, y_test), callbacks=[early_stop], verbose=verbose)

        # save deep learning model
        model_filepath = model_path + self.model_name + '_model.keras'
        model.save(model_filepath)

        # return model, X_test_pre, y_test, encoder, scaler, history

        if show_eval:
            self.plot_rnn_training_eval(history)
            self.eval_classification_model(model, X_test_pre, y_test, one_hot_encoder=encoder)

        return model

    # def train_dl_trend_prediction_model(self,feature_function='get_features_v1', test_size=0.2, epochs=300):
    #     # 1. Get features and target
    #     # Get features
    #     X = df.copy()
    #     X = globals()[feature_function](X, T=T)
    #     X.dropna(inplace=True)

    #     # Get target
    #     y = df.copy()
    #     y = self.get_target(y, target_shift = self.predict_period)
    #     y.dropna(inplace=True)

    #     # 2. Split into train and test sets
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    #     # 3. Scale the data
    #     scaler = MinMaxScaler()
    #     X_train_scaled = scaler.fit_transform(X_train)
    #     X_test_scaled = scaler.transform(X_test)

    #     # 4. Reshape the data
    #     X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], T, 1)
    #     X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], T, 1)

    #     # 5. One-hot encode the target
    #     encoder = OneHotEncoder(sparse=False)
    #     y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
    #     y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1))

    #     # 6. Build the model
    #     model = Sequential()
    #     model.add(LSTM(50, input_shape=(T, 1)))
    #     model.add(Dense(self.C, activation='softmax'))
    #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #     # 7. Fit the model
    #     model.fit(X_train_scaled, y_train_encoded, epochs=epochs, validation_data=(X_test_scaled, y_test_encoded), verbose=1)

    #     # 8. Save the model, scaler and encoder
    #     model.save(f"./models/{model_name}.h5")
    #     joblib.dump(scaler, f"./models/{model_name}_scaler.joblib")
    #     joblib.dump(encoder, f"./models/{model_name}_encoder.joblib")

    #     return model

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
# predictor = BtcTrendPredictor(predict_period = 1)


# == Train ==
# df = predictor.get_training_data()
# model_path =  predictor.train_dl_trend_prediction_model(test_size=0.2, epochs=30)


# == Predict ==

# df = predictor.fetch_live_binance_data()
# print(df)

# df = predictor.predict()
# print(df)

# df = predictor.get_target(df)
# print(df)