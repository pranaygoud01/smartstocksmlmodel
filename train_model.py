{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0816c1fd-c5c5-4e0a-b25b-2fbbff7bef17",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting alpha_vantage\n",
      "  Downloading alpha_vantage-3.0.0-py3-none-any.whl.metadata (12 kB)\n",
      "Collecting aiohttp (from alpha_vantage)\n",
      "  Downloading aiohttp-3.10.10-cp311-cp311-win_amd64.whl.metadata (7.8 kB)\n",
      "Requirement already satisfied: requests in c:\\users\\prana\\appdata\\local\\packages\\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\\localcache\\local-packages\\python311\\site-packages (from alpha_vantage) (2.32.3)\n",
      "Collecting aiohappyeyeballs>=2.3.0 (from aiohttp->alpha_vantage)\n",
      "  Downloading aiohappyeyeballs-2.4.3-py3-none-any.whl.metadata (6.1 kB)\n",
      "Collecting aiosignal>=1.1.2 (from aiohttp->alpha_vantage)\n",
      "  Downloading aiosignal-1.3.1-py3-none-any.whl.metadata (4.0 kB)\n",
      "Requirement already satisfied: attrs>=17.3.0 in c:\\users\\prana\\appdata\\local\\packages\\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\\localcache\\local-packages\\python311\\site-packages (from aiohttp->alpha_vantage) (24.2.0)\n",
      "Collecting frozenlist>=1.1.1 (from aiohttp->alpha_vantage)\n",
      "  Downloading frozenlist-1.5.0-cp311-cp311-win_amd64.whl.metadata (14 kB)\n",
      "Collecting multidict<7.0,>=4.5 (from aiohttp->alpha_vantage)\n",
      "  Downloading multidict-6.1.0-cp311-cp311-win_amd64.whl.metadata (5.1 kB)\n",
      "Collecting yarl<2.0,>=1.12.0 (from aiohttp->alpha_vantage)\n",
      "  Downloading yarl-1.17.1-cp311-cp311-win_amd64.whl.metadata (66 kB)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\prana\\appdata\\local\\packages\\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\\localcache\\local-packages\\python311\\site-packages (from requests->alpha_vantage) (3.4.0)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\prana\\appdata\\local\\packages\\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\\localcache\\local-packages\\python311\\site-packages (from requests->alpha_vantage) (3.10)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\prana\\appdata\\local\\packages\\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\\localcache\\local-packages\\python311\\site-packages (from requests->alpha_vantage) (2.2.3)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\prana\\appdata\\local\\packages\\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\\localcache\\local-packages\\python311\\site-packages (from requests->alpha_vantage) (2024.8.30)\n",
      "Collecting propcache>=0.2.0 (from yarl<2.0,>=1.12.0->aiohttp->alpha_vantage)\n",
      "  Downloading propcache-0.2.0-cp311-cp311-win_amd64.whl.metadata (7.9 kB)\n",
      "Downloading alpha_vantage-3.0.0-py3-none-any.whl (35 kB)\n",
      "Downloading aiohttp-3.10.10-cp311-cp311-win_amd64.whl (381 kB)\n",
      "Downloading aiohappyeyeballs-2.4.3-py3-none-any.whl (14 kB)\n",
      "Downloading aiosignal-1.3.1-py3-none-any.whl (7.6 kB)\n",
      "Downloading frozenlist-1.5.0-cp311-cp311-win_amd64.whl (51 kB)\n",
      "Downloading multidict-6.1.0-cp311-cp311-win_amd64.whl (28 kB)\n",
      "Downloading yarl-1.17.1-cp311-cp311-win_amd64.whl (90 kB)\n",
      "Downloading propcache-0.2.0-cp311-cp311-win_amd64.whl (44 kB)\n",
      "Installing collected packages: propcache, multidict, frozenlist, aiohappyeyeballs, yarl, aiosignal, aiohttp, alpha_vantage\n",
      "Successfully installed aiohappyeyeballs-2.4.3 aiohttp-3.10.10 aiosignal-1.3.1 alpha_vantage-3.0.0 frozenlist-1.5.0 multidict-6.1.0 propcache-0.2.0 yarl-1.17.1\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install alpha_vantage\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7b371867-139d-4fef-b313-af2d513a0d6c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\prana\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\keras\\src\\layers\\rnn\\rnn.py:204: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(**kwargs)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m8s\u001b[0m 29ms/step - loss: 0.0042\n",
      "Epoch 2/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m7s\u001b[0m 33ms/step - loss: 0.0010\n",
      "Epoch 3/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 31ms/step - loss: 5.6647e-04\n",
      "Epoch 4/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 30ms/step - loss: 5.2297e-04\n",
      "Epoch 5/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 32ms/step - loss: 2.6654e-04\n",
      "Epoch 6/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 31ms/step - loss: 3.2917e-04\n",
      "Epoch 7/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 30ms/step - loss: 3.8611e-04\n",
      "Epoch 8/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 31ms/step - loss: 3.5318e-04\n",
      "Epoch 9/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 31ms/step - loss: 2.5290e-04\n",
      "Epoch 10/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m9s\u001b[0m 27ms/step - loss: 2.2892e-04\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['scaler.pkl']"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from alpha_vantage.timeseries import TimeSeries\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import LSTM, Dense\n",
    "import joblib\n",
    "import os\n",
    "\n",
    "# Alpha Vantage API setup\n",
    "API_KEY = \"Q6TVUD7EYBHCZ689\"\n",
    "ts = TimeSeries(key=API_KEY, output_format='pandas')\n",
    "\n",
    "# Fetch data for a specific stock symbol (e.g., 'AAPL')\n",
    "symbol = 'AAPL'\n",
    "data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')\n",
    "data = data[['4. close']].rename(columns={'4. close': 'Close'})  # Use 'Close' prices\n",
    "data = data.iloc[::-1]  # Reverse data to chronological order\n",
    "\n",
    "# Preprocess the data\n",
    "scaler = MinMaxScaler(feature_range=(0, 1))\n",
    "scaled_data = scaler.fit_transform(data)\n",
    "\n",
    "# Create sequences\n",
    "def create_sequences(data, seq_length):\n",
    "    x, y = [], []\n",
    "    for i in range(seq_length, len(data)):\n",
    "        x.append(data[i-seq_length:i, 0])\n",
    "        y.append(data[i, 0])\n",
    "    return np.array(x), np.array(y)\n",
    "\n",
    "seq_length = 60  # Example sequence length\n",
    "x, y = create_sequences(scaled_data, seq_length)\n",
    "x = np.reshape(x, (x.shape[0], x.shape[1], 1))\n",
    "\n",
    "# Build the LSTM model\n",
    "model = Sequential([\n",
    "    LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)),\n",
    "    LSTM(units=50, return_sequences=False),\n",
    "    Dense(units=25),\n",
    "    Dense(units=1)\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam', loss='mean_squared_error')\n",
    "model.fit(x, y, epochs=10, batch_size=32)\n",
    "\n",
    "# Save the model and scaler\n",
    "model.save(\"lstm_stock_model.h5\")\n",
    "joblib.dump(scaler, \"scaler.pkl\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a05aa0d9-368f-4d5e-87cb-f1724a820575",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m8s\u001b[0m 29ms/step - loss: 0.0071\n",
      "Epoch 2/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 27ms/step - loss: 9.5011e-04\n",
      "Epoch 3/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 28ms/step - loss: 8.0189e-04\n",
      "Epoch 4/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m7s\u001b[0m 33ms/step - loss: 8.4767e-04\n",
      "Epoch 5/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m7s\u001b[0m 33ms/step - loss: 3.0908e-04\n",
      "Epoch 6/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m7s\u001b[0m 34ms/step - loss: 1.9740e-04\n",
      "Epoch 7/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m12s\u001b[0m 41ms/step - loss: 3.7704e-04\n",
      "Epoch 8/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 32ms/step - loss: 4.4518e-04\n",
      "Epoch 9/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m7s\u001b[0m 35ms/step - loss: 4.4915e-04\n",
      "Epoch 10/10\n",
      "\u001b[1m195/195\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m9s\u001b[0m 27ms/step - loss: 6.6896e-04\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['scaler.pkl']"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from tensorflow.keras.layers import Input, LSTM, Dense\n",
    "from tensorflow.keras.models import Sequential\n",
    "import joblib\n",
    "\n",
    "# Model Definition\n",
    "model = Sequential([\n",
    "    Input(shape=(x.shape[1], 1)),  # Updated input layer\n",
    "    LSTM(units=50, return_sequences=True),\n",
    "    LSTM(units=50, return_sequences=False),\n",
    "    Dense(units=25),\n",
    "    Dense(units=1)\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam', loss='mean_squared_error')\n",
    "model.fit(x, y, epochs=10, batch_size=32)\n",
    "\n",
    "# Save the model in the recommended format\n",
    "model.save(\"lstm_stock_model.keras\")  # Changed to .keras format\n",
    "joblib.dump(scaler, \"scaler.pkl\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9c257c29-57b0-4c64-ae75-9a5ad99bcc5d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
