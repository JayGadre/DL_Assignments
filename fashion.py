import tensorflow as tf
from tensorflow.keras import models,layers
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
(X_train,y_train),(X_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()


X_train = X_train/255.0
X_test = X_test/255.0

model = models.Sequential([
    layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPool2D((3,3)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPool2D((3,3)),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(X_train,y_train,epochs=5,validation_data=(X_test,y_test))

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)

print(accuracy_score(y_test,y_pred))