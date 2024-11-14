#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import tensorflow as tf
import joblib
from alpha_vantage.timeseries import TimeSeries
import os
API_KEY = os.getenv('ALPHA_API_KEY')  # Set a default if necessary

app = Flask(__name__)
# Allow requests from 'http://localhost:3000' only
CORS(app, origins="https://smartstocks.vercel.app/")

# Load model and scaler
model = tf.keras.models.load_model("lstm_stock_model.keras")
model.compile(optimizer='adam', loss='mean_squared_error')
scaler = joblib.load("scaler.pkl")

# Alpha Vantage API setup

ts = TimeSeries(key=API_KEY, output_format='pandas')


# Return a 500 status with the error message
@app.route('/predict', methods=['POST'])
def predict():
    try:
        symbol = request.json.get('symbol', 'AAPL')
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400

        # Fetch stock data
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
        if data.empty:
            return jsonify({'error': f'No data found for symbol: {symbol}'}), 404

        data = data[['4. close']].rename(columns={'4. close': 'Close'}).iloc[::-1]
        scaled_data = scaler.transform(data[-60:].values)
        x = np.array([scaled_data])

        # Make prediction
        prediction = model.predict(x)
        prediction = scaler.inverse_transform(prediction)

        # Convert numpy float32 to Python float
        prediction_value = float(prediction[0][0])

        return jsonify({'symbol': symbol, 'prediction': prediction_value})

    except Exception as e:
        # Return a 500 status with the error message
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)


# In[2]:





# In[ ]:




