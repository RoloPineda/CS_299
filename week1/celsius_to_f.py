import tensorflow as tf
import numpy as np
from tensorflow import keras

# This regression ML model converts Celsius degrees to fahrenheit
celsius = np.array([0, 16, 20, 50, 100, 42], dtype=float)
fahrenheit = np.array([32, 60.8, 68, 122, 212, 107.6], dtype=float)

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')
history = model.fit(celsius, fahrenheit, epochs=1500)

print("Predicting...", model.predict([10]))
