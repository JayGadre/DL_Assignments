import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np


ticker = "AAPL"
data = yf.download(ticker, start="2010-01-01", end="2024-01-01")
data = data[["Close"]]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

LOOKBACK_DAYS = 60
X, y = [], []

for i in range(LOOKBACK_DAYS, len(scaled_data)):
    X.append(scaled_data[i-LOOKBACK_DAYS:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# RNNs strictly require a 3D input shape: (samples, timesteps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into 80% Training and 20% Testing data
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


model = Sequential([
    # First LSTM Layer. return_sequences=False because the next layer is Dense
    LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)),
    
    # Standard Dense layer to compute the final output price
    Dense(25),
    Dense(1) 
])

model.compile(optimizer='adam', loss='mean_squared_error')

print("\n--- Training the RNN ---")
model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))

last_60_days = scaled_data[-LOOKBACK_DAYS:]
last_60_days = np.reshape(last_60_days, (1, LOOKBACK_DAYS, 1))

# Predict and reverse the 0-1 scaling back to actual dollars
predicted_price_scaled = model.predict(last_60_days)
predicted_price = scaler.inverse_transform(predicted_price_scaled)

print(f"\n🚀 Predicted next day closing price for {ticker}: ${predicted_price[0][0]:.2f}")

accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {accuracy:.6f}")