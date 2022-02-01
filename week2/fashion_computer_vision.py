import matplotlib.pyplot as plt
import tensorflow as tf

print(tf.__version__)
import numpy as np
import pandas as pd

# downloading data set
fashion_mnist = tf.keras.datasets.fashion_mnist

# Splitting data into testing and training subsets
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

index = 42

# set number of characters per row when printing
np.set_printoptions(linewidth=320)

# Print the label and image
print(f'Label: {training_labels[index]}')
print(f'\nImage Pixel Array:\n {training_images[index]}')

# visualize the image
plt.imshow(training_images[index])
