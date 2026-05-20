import tensorflow as tf
from tensorflow.keras import layers,models
import numpy as np


fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()    

X_train_images = X_train/255.0
X_test_images = X_test/255.0

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape = (28,28,1)),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPool2D((2,2)),

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

model.fit(
    X_train_images,
    y_train,
    epochs=10,
    validation_data=(X_test_images,y_test)
)

test_loss,test_acc = model.evaluate(X_test_images,y_test)
print('Test accuracy:',test_acc)