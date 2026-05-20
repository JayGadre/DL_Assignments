import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Generate 1000 days of synthetic temperatures mimicking summer/winter waves
# Base temperature drops and climbs cleanly across seasons
days = np.arange(1000)
seasonal_trend = 20 + 15 * np.sin(2 * np.pi * days / 365) 
noise = np.random.normal(0, 2, 1000) # Adds daily random weather variations
temperatures = seasonal_trend + noise

# 2. Slice data into rolling 14-day sequence windows
LOOKBACK = 14
X, y = [], []
for i in range(LOOKBACK, len(temperatures)):
    X.append(temperatures[i-LOOKBACK:i])
    y.append(temperatures[i])

X, y = np.array(X), np.array(y)
X = np.expand_dims(X, -1)  # Reshape to 3D for LSTM: (samples, timesteps, 1)

# 3. Design and train a lightweight LSTM network
model = Sequential([
    LSTM(32, activation='relu', input_shape=(LOOKBACK, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

print("--- Training Weather Model ---")
model.fit(X, y, epochs=10, batch_size=32, verbose=1) # verbose=1 lets you monitor loss drops!

# 4. Predict tomorrow's temperature using the most recent 14 days
last_14_days = np.array([temperatures[-LOOKBACK:]]).reshape(1, LOOKBACK, 1)
tomorrow_prediction = model.predict(last_14_days, verbose=0)[0][0]

print(f"\n📊 Past 3 days simulated temps: {np.round(temperatures[-3:], 1)}")
print(f"🔮 LSTM Predicted temperature for tomorrow: {tomorrow_prediction:.1f}°")

accuracy = model.evaluate(X, y, verbose=0)
print(f"Model Training Loss (MSE): {accuracy:.4f}")