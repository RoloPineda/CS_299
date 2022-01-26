import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib
from matplotlib import pyplot as pl

## This is a simple neural network that solves the linear equation y=8x-9

input_data = np.array([1.0, 2.0, 5.0, 100.0, 15.0, 22.0, 10.0, 7.0, 11, 54], dtype=float)
output_data = np.array([-1.0, 7.0, 31, 791.0, 111.0, 167.0, 71, 47, 79, 423], dtype=float)

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
opt = keras.optimizers.Adam(learning_rate=0.09)
model.compile(optimizer=opt, loss='mean_squared_error')

model.fit(input_data, output_data, epochs=1700)

# observation: Model seems to perform better utilizing a regression optimizer(Adam) as opposed to sgd
# It also performs better on numbers<150
print("prediction: ", (model.predict([150])))
