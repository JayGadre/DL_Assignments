import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
import seaborn as sns

# 1. Load Data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Reshape & Normalize
X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))
X_train, X_test = X_train / 255.0, X_test / 255.0

# 3. Build CNN Architecture
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 4. Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train Model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# 6. Predict & Convert Probabilities to Class Labels
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1) # <-- The crucial fix

# 7. Evaluate
print(classification_report(y_test, y_pred))