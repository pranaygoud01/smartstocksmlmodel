

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import tensorflow as tf
import joblib
from alpha_vantage.timeseries import TimeSeries
import os
API_KEY = os.getenv('ALPHA_API_KEY')  # Set a default if necessary



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


app = Flask(__name__)


CORS(app)

# Load model and scaler
model = tf.keras.models.load_model("lstm_stock_model.keras")
model.compile(optimizer='adam', loss='mean_squared_error')
scaler = joblib.load("scaler.pkl")

# Alpha Vantage API setup

ts = TimeSeries(key=API_KEY, output_format='pandas')
@app.route('/')
def home():
    return 'Flask Server is Running'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symbol = request.json.get('symbol', 'AAPL').upper()
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400

        # Fetch stock data
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
        if data.empty:
            return jsonify({'error': f'No data found for symbol: {symbol}'}), 404

        # Process the data
        data = data[['4. close']].rename(columns={'4. close': 'Close'}).iloc[::-1]
        if len(data) < 60:
            return jsonify({'error': 'Not enough data for prediction'}), 400

        scaled_data = scaler.transform(data[-60:].values)
        x = np.array([scaled_data])

        # Make prediction
        prediction = model.predict(x)
        prediction_value = float(scaler.inverse_transform(prediction)[0][0])

        return jsonify({'symbol': symbol, 'prediction': prediction_value})

    except Exception as e:
        # Log error for debugging
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

    except Exception as e:
        # Return a 500 status with the error message
        return jsonify({'error': str(e)}), 500



# Allow requests from 'http://localhost:3000' only
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)