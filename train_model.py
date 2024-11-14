#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install alpha_vantage


# In[2]:


import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os

# Alpha Vantage API setup
API_KEY = "Q6TVUD7EYBHCZ689"
ts = TimeSeries(key=API_KEY, output_format='pandas')

# Fetch data for a specific stock symbol (e.g., 'AAPL')
symbol = 'AAPL'
data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
data = data[['4. close']].rename(columns={'4. close': 'Close'})  # Use 'Close' prices
data = data.iloc[::-1]  # Reverse data to chronological order

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

seq_length = 60  # Example sequence length
x, y = create_sequences(scaled_data, seq_length)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)),
    LSTM(units=50, return_sequences=False),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, epochs=10, batch_size=32)

# Save the model and scaler
model.save("lstm_stock_model.h5")
joblib.dump(scaler, "scaler.pkl")


# In[3]:


from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Sequential
import joblib

# Model Definition
model = Sequential([
    Input(shape=(x.shape[1], 1)),  # Updated input layer
    LSTM(units=50, return_sequences=True),
    LSTM(units=50, return_sequences=False),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, epochs=10, batch_size=32)

# Save the model in the recommended format
model.save("lstm_stock_model.keras")  # Changed to .keras format
joblib.dump(scaler, "scaler.pkl")


# In[ ]:




