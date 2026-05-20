import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,models

data_dir = "PlantVillage"
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(128,128)
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(128,128)
)

class_names = train_ds.class_names
print(class_names)

model = models.Sequential([
    layers.Rescaling(1.0/255.0,input_shape=(128,128,3)),
    layers.Conv2D(32,3,activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64,3,activation='relu'),
    layers.MaxPool2D(),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(len(class_names),activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_ds,epochs=5)

loss,accuracy = model.evaluate(test_ds)

print(accuracy)