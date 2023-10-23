
import pandas as pd
import numpy as np
import joblib
import requests
import json
import datetime
import talib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2

import matplotlib.pyplot as plt
import seaborn as sns
import time
import mplfinance as mpf


class BtcTrendPredictor:
    
    def __init__(self, predict_period = 1, model_path = './models/', plot_path = './predict_live_plot/'):
        
        self.T = 72 # number of periods to use as input features (time steps)
        self.predict_period = predict_period # number of periods to predict
        self.C = 6 # number of classes
        self.model_path = model_path
        self.model_name = f'btc_live_pred_{predict_period}hr'
        self.plot_path = plot_path

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

    def predict_live_data(self, df_live, feature_function='get_features_v1'):
        # Get features
        X_live = df_live.copy()

        # Call function in variable feature_function (get_features_v1)
        feature_to_call = getattr(self, feature_function)

        if callable(feature_to_call):
            X_live = feature_to_call(X_live, T=self.T)
        else:
            print(f"{feature_function} is not a valid method of this class.")

        X_live.dropna(inplace=True)

        # Dimension parameters
        F = int((X_live.shape[1]) / self.T) # number of input features 
        # print(f"Input shape: ({self.T}, {F})")

        # load scaler
        scaler_filepath = self.model_path + self.model_name + '_scaler.joblib'
        scaler = joblib.load(scaler_filepath)
        X_live_pre = scaler.transform(X_live)

        X_live_pre = X_live_pre.reshape(-1, self.T, F)

        # load deep learning model
        model_filepath = self.model_path + self.model_name + '_model.keras'
        model = tf.keras.models.load_model(model_filepath)

        # predict
        y_pred = model.predict(X_live_pre, verbose=0)

        # load encoder
        encoder_filepath = self.model_path + self.model_name + '_encoder.joblib'
        encoder = joblib.load(encoder_filepath)

        # reverse one-hot encoding
        y_pred = encoder.inverse_transform(y_pred)
        y_pred = y_pred.flatten()

        # insert missing prediction with nan
        len_diff = len(df_live) - len(y_pred)
        y_pred = np.insert(y_pred, 0,  [np.nan] * len_diff, axis=0)

        return y_pred
    
    def evaluate_live_data(self, df_live, y_live_pred, to_file=True):
        # Get prediction result
        # y_live_pred = self.predict_live_data(df_live, feature_function=feature_function)

        pred_result = df_live.copy()
        pred_result.loc[:, 'prediction'] = y_live_pred

        # Get target
        pred_result = self.get_target(pred_result, target_shift = self.predict_period)
        pred_result.dropna(inplace=True)

        y_test = pred_result['target']
        y_pred = pred_result['prediction']
        
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

        # Calculate expired time from last datetime in df_live index + 1 hour and convert to epoch
        # last_datetime = df_live.index[-1] + datetime.timedelta(hours=2) # 1 hr offset + 1 hr API update interval
        # expired_time = int(last_datetime.timestamp())
        # print(f"Expired time: {last_datetime}, {expired_time}")


        plot_filepath = self.plot_path + self.model_name + '_eval.png'

        if to_file:
            # save plot to file
            fig.savefig(plot_filepath)
        else:
            plt.tight_layout()
            plt.show()

        return plot_filepath

    def plot_live_trend_prediction(self, df_live, y_live_pred, to_file=True):

        # Get prediction result
        # y_live_pred = predict_live_data(df_live, T, model_name = model_name, model_path = model_path, feature_function=feature_function)
        
        pred_result = df_live.copy()
        pred_result.loc[:, 'prediction'] = y_live_pred

        # ohlc
        pred_result['ohlc'] = (pred_result['open'] + pred_result['high'] + pred_result['low'] + pred_result['close']) / 4

        # diff between ema 10 and 20
        pred_result['ema_10'] = talib.EMA(pred_result['ohlc'], timeperiod=10)
        pred_result['ema_20'] = talib.EMA(pred_result['ohlc'], timeperiod=20)

        pred_result.dropna(inplace=True)

        marker_space = 200

        # Initialize the marker columns with NaNs
        pred_result['prediction_price_marker_p3'] = np.nan
        pred_result['prediction_price_marker_p2'] = np.nan
        pred_result['prediction_price_marker_p1'] = np.nan
        pred_result['prediction_price_marker_m3'] = np.nan
        pred_result['prediction_price_marker_m2'] = np.nan
        pred_result['prediction_price_marker_m1'] = np.nan

        pred_result.loc[pred_result['prediction'] == 3, 'prediction_price_marker_p3'] = pred_result['low'] - marker_space
        pred_result.loc[pred_result['prediction'] == 2, 'prediction_price_marker_p2'] = pred_result['low'] - marker_space
        pred_result.loc[pred_result['prediction'] == 1, 'prediction_price_marker_p1'] = pred_result['low'] - marker_space

        pred_result.loc[pred_result['prediction'] == -3, 'prediction_price_marker_m3'] = pred_result['high'] + marker_space
        pred_result.loc[pred_result['prediction'] == -2, 'prediction_price_marker_m2'] = pred_result['high'] + marker_space
        pred_result.loc[pred_result['prediction'] == -1, 'prediction_price_marker_m1'] = pred_result['high'] + marker_space

        pred_result['prediction_ind_marker_p3'] = np.nan
        pred_result['prediction_ind_marker_p2'] = np.nan
        pred_result['prediction_ind_marker_p1'] = np.nan
        pred_result['prediction_ind_marker_m3'] = np.nan
        pred_result['prediction_ind_marker_m2'] = np.nan
        pred_result['prediction_ind_marker_m1'] = np.nan

        pred_result.loc[pred_result['prediction'] == 3, 'prediction_ind_marker_p3'] = pred_result['ema_20'] - marker_space
        pred_result.loc[pred_result['prediction'] == 2, 'prediction_ind_marker_p2'] = pred_result['ema_20'] - marker_space
        pred_result.loc[pred_result['prediction'] == 1, 'prediction_ind_marker_p1'] = pred_result['ema_20'] - marker_space

        pred_result.loc[pred_result['prediction'] == -3, 'prediction_ind_marker_m3'] = pred_result['ema_20'] + marker_space
        pred_result.loc[pred_result['prediction'] == -2, 'prediction_ind_marker_m2'] = pred_result['ema_20'] + marker_space
        pred_result.loc[pred_result['prediction'] == -1, 'prediction_ind_marker_m1'] = pred_result['ema_20'] + marker_space


        # Plot
        # Set seaborn style
        sns.set_style("whitegrid")

        # Create subplots: one for price, one for MACD histogram
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(25,25), gridspec_kw={'height_ratios': [2, 1, 2]})
        fig.tight_layout(pad=3.0)

        # Plotting price data on the first subplot
        axes[0].plot(pred_result['close'], label='Close Price', color='blue', linewidth=1)

        axes[0].scatter(
                pred_result['prediction_price_marker_p1'].index, 
                pred_result['prediction_price_marker_p1'],
                color='greenyellow', 
                marker='.', 
                label='Prediction = + Weak',
                s=10,
            )

        axes[0].scatter(
                pred_result['prediction_price_marker_p2'].index, 
                pred_result['prediction_price_marker_p2'],
                color='limegreen', 
                marker='^', 
                label='Prediction = +',
                s=20
            )
        
        axes[0].scatter(
                pred_result['prediction_price_marker_p3'].index, 
                pred_result['prediction_price_marker_p3'],
                color='darkgreen', 
                marker='^', 
                label='Prediction = + Strong',
                s=20
            )

        axes[0].scatter(
                pred_result['prediction_price_marker_m1'].index, 
                pred_result['prediction_price_marker_m1'],
                color='lightsalmon',
                marker='.', 
                label='Prediction = - Weak',
                s=10,
            )

        axes[0].scatter(
                pred_result['prediction_price_marker_m2'].index, 
                pred_result['prediction_price_marker_m2'],
                color='coral',
                marker='v', 
                label='Prediction = -',
                s=20,
            )

        axes[0].scatter(
                pred_result['prediction_price_marker_m3'].index, 
                pred_result['prediction_price_marker_m3'],
                color='red',
                marker='v', 
                label='Prediction = - Strong',
                s=20,
            )
        
        axes[0].set_title(f'BTCUSDT Trend Prediction {self.predict_period} hr ahead')
        axes[0].set_ylabel('Price')
        axes[0].legend(loc='lower right')

        # Filling in the area between high and low prices (optional)
        axes[0].fill_between(pred_result.index, pred_result['high'], pred_result['low'], color='skyblue', alpha=0.6)

        # line plot EMA 10 and 20
        axes[1].plot(pred_result.index, pred_result['ema_10'], label='EMA 10', color='orange', alpha=0.7, linewidth=1)
        axes[1].plot(pred_result.index, pred_result['ema_20'], label='EMA 20', color='slateblue', alpha=0.7, linewidth=1)

        axes[1].scatter(
                pred_result['prediction_ind_marker_p1'].index, 
                pred_result['prediction_ind_marker_p1'],
                color='greenyellow', 
                marker='.', 
                s=10,
            )

        axes[1].scatter(
                pred_result['prediction_ind_marker_p2'].index, 
                pred_result['prediction_ind_marker_p2'],
                color='limegreen', 
                marker='^', 
                # label='Prediction = +',
                s=20
            )
        
        axes[1].scatter(
                pred_result['prediction_ind_marker_p3'].index, 
                pred_result['prediction_ind_marker_p3'],
                color='darkgreen', 
                marker='^', 
                # label='Prediction = + Strong',
                s=20
            )

        axes[1].scatter(
                pred_result['prediction_ind_marker_m1'].index, 
                pred_result['prediction_ind_marker_m1'],
                color='lightsalmon',
                marker='.', 
                s=10,
            )

        axes[1].scatter(
                pred_result['prediction_ind_marker_m2'].index, 
                pred_result['prediction_ind_marker_m2'],
                color='lightcoral',
                marker='v', 
                s=20,
            )
        
        axes[1].scatter(
                pred_result['prediction_ind_marker_m3'].index, 
                pred_result['prediction_ind_marker_m3'],
                color='red',
                marker='v', 
                s=20,
        )

        axes[1].set_title('EMAs Trend prediction')
        axes[1].set_ylabel('Value')
        axes[1].set_xlabel('Period')
        axes[1].legend(loc='lower right')

        # plot candlestick
        df_candle = pred_result.iloc[-48:].copy()
        df_candle.index = pd.to_datetime(df_candle.index)

        
        # ====
        # Use mplfinance to plot on axes[2]
        # add_plot = [mpf.make_addplot(df_candle['prediction_price_marker_p1'], marker='.', color='greenyellow', markersize=100)]
        add_plot = [
            # plot EMA 10 and 20
            mpf.make_addplot(df_candle['ema_10'], ax=axes[2], color='orange', alpha=0.7),
            mpf.make_addplot(df_candle['ema_20'], ax=axes[2], color='slateblue', alpha=0.7),

            mpf.make_addplot(df_candle['prediction_price_marker_p1'], ax=axes[2], type='scatter', marker='.', color='greenyellow', markersize=50),
            mpf.make_addplot(df_candle['prediction_price_marker_p2'], ax=axes[2], type='scatter', marker='^', color='limegreen', markersize=50),
            mpf.make_addplot(df_candle['prediction_price_marker_p3'], ax=axes[2], type='scatter', marker='^', color='darkgreen', markersize=50),
            mpf.make_addplot(df_candle['prediction_price_marker_m1'], ax=axes[2], type='scatter', marker='.', color='lightsalmon', markersize=50),
            mpf.make_addplot(df_candle['prediction_price_marker_m2'], ax=axes[2], type='scatter', marker='v', color='coral', markersize=50),
            mpf.make_addplot(df_candle['prediction_price_marker_m3'], ax=axes[2], type='scatter', marker='v', color='red', markersize=50),
        ]
        mpf.plot(df_candle, type='candle', ax=axes[2], addplot=add_plot, style='yahoo')

        # Adjust the x-axis of the candlestick plot
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].xaxis.set_major_locator(plt.MaxNLocator(nbins=len(df_candle)))
        axes[2].set_xticks(range(0, len(df_candle.index), 1))
        axes[2].set_xticklabels(df_candle.index, rotation=90)

        # set plot title
        axes[2].set_title(f'BTCUSDT Trend Prediction {self.predict_period} hr ahead- Candlestick')

        plot_filepath = self.plot_path + self.model_name + '_prediction.png'
        if to_file:
            # save plot to file
            fig.savefig(plot_filepath)
        else:
            plt.show()

        return plot_filepath

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