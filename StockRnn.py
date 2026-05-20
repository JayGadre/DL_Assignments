import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models

# =====================================================================
# STEP 1: Load Real Stock Price Data from public URL
# =====================================================================
url = "https://raw.githubusercontent.com/plotly/datasets/master/stockdata.csv"
df = pd.read_csv(url)

# This dataset contains multiple tech stocks; we will extract just 'GOOG' (Google)
data = df["GOOG"].values.reshape(-1, 1)

# Normalize data between 0 and 1 (highly recommended for RNN stability)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# =====================================================================
# STEP 2: Create a Time Window (Lookback)
# =====================================================================
# Look at the past 5 days of stock prices to predict the 6th day
lookback = 5
X, y = [], []

for i in range(len(data) - lookback):
    X.append(data[i : i + lookback])
    y.append(data[i + lookback])

X, y = np.array(X), np.array(y)

# Split into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================================
# STEP 3: Build and Train the SimpleRNN Model
# =====================================================================
model = models.Sequential(
    [
        # Swap out 'LSTM' for 'SimpleRNN' as per assignment requirement
        layers.SimpleRNN(64, activation="relu", input_shape=(lookback, 1)),
        layers.Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mse")
print("Training the RNN model...")
model.fit(X_train, y_train, epochs=10, batch_size=16)

# =====================================================================
# STEP 4: Make Predictions and De-scale
# =====================================================================
y_pred = model.predict(X_test)

# Reverse data scaling back into actual stock prices (USD)
y_pred_actual = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

print("\nModel Evaluation:")
print("R2 Score:", round(r2_score(y_test_actual, y_pred_actual), 4))

# =====================================================================
# STEP 5: Visualizing the Results
# =====================================================================
plt.figure(figsize=(10, 5))
plt.plot(y_test_actual[:50], color="steelblue", label="Actual Google Price")
plt.plot(
    y_pred_actual[:50],
    color="darkorange",
    linestyle="--",
    label="RNN Predicted Price",
)
plt.title("Google Stock Price Prediction using SimpleRNN")
plt.xlabel("Sample Days")
plt.ylabel("Stock Price ($)")
plt.legend()
plt.show()