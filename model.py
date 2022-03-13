import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(400, 400, 3))
rescale = layers.Rescaling(1./255)
conv1 = layers.Conv2D(2, 3, activation='relu')
flatten = layers.Flatten()
dense = layers.Dense(100, activation='relu')
dense2 = layers.Dense(25, activation='relu')
dense3 = layers.Dense(2, activation='sigmoid')

x = rescale(inputs)
x = conv1(x)
x = flatten(x)
x = dense(x)
x = dense2(x)
x = dense3(x)


model = keras.Model(inputs=inputs, outputs=x)

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

model.save("model")
