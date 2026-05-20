import tensorflow as tf
from tensorflow.keras import layers,models
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

(X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train/255.0
x_test = X_test/255.0

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPool2D((3,3)),
    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPool2D((3,3)),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train,y_train,epochs = 5,validation_data=(X_test,y_test))

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis = 1)

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))