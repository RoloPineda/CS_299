import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
#plt.imshow(training_images[0])
#plt.show()
#print(training_labels[0])
#print(training_images[0])

training_images = training_images / 255.0
test_images = test_images/255.0
