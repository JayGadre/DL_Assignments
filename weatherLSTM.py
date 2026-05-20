from tensorflow.keras import layers,models
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"

df = pd.read_csv(url)

data = df['Temp'].values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)
lookback = 3
X,y=[],[]

for i in range(len(data)-lookback):
    X.append(data[i:i+lookback])
    y.append(data[i+lookback])

X,y=np.array(X),np.array(y)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = models.Sequential([
    layers.LSTM((512),activation='relu',input_shape=(lookback,1)),
    layers.Dense(1)
])

model.compile(optimizer='adam',loss='mse')

model.fit(X_train,y_train,epochs=20)

y_pred = model.predict(X_test)

y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)


print(r2_score(y_test,y_pred))


