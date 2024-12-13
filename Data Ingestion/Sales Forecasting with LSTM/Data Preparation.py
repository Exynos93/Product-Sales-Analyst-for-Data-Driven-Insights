import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load and preprocess data
df = pd.read_csv('sales_data.csv', parse_dates=['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_promotion'] = df['promotion'].apply(lambda x: 1 if x > 0 else 0)  # 1 if there's a promotion, 0 otherwise
df['lag_1'] = df['sales'].shift(1)  # Lag features for time series
df['rolling_mean'] = df['sales'].rolling(window=7).mean()  # Rolling mean

# Drop NaN values created by lag and rolling features
df = df.dropna()

# Select features
features = ['sales', 'lag_1', 'rolling_mean', 'day_of_week', 'month', 'is_promotion']
target = 'sales'

# Normalize features
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_x.fit_transform(df[features])
scaled_target = scaler_y.fit_transform(df[target].values.reshape(-1, 1))

# Prepare sequences for LSTM (assuming we use last 10 days to predict next day)
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length][0]  # Predict next day sales
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(scaled_features, seq_length)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], seq_length, len(features)))
X_test = np.reshape(X_test, (X_test.shape[0], seq_length, len(features)))
