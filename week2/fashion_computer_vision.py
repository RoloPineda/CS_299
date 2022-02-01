import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
import numpy as np
import pandas as pd

# downloading data set
fashion_mnist = tf.keras.datasets.fashion_mnist

# Splitting data into testing and training subsets
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

index = 0

# set number of characters per row when printing
np.set_printoptions(linewidth=320)

# Print the label and image
print(f'Label: {training_labels[index]}')
print(f'\nImage Pixel Array:\n {training_images[index]}')

# visualize the image
plt.imshow(training_images[index])
plt.show()

# Normalize the pixel values of the training and testing images
training_images = training_images / 255.0
test_images = test_images / 255.0

# relu passes values greater than 0 to the next layer
# softmax takes a list of values and scales them so the sum of all elements will be equal to 1
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Declare sample inputs and convert to a tensor
inputs = np.array([[1.0, 3.0, 4.0, 2.0]])
inputs = tf.convert_to_tensor(inputs)
print(f'input to softmax function: {inputs.numpy()}')

# Feed the inputs to a softmax activation function
outputs = tf.keras.activations.softmax(inputs)
print(f'Output of softmax function: {outputs.numpy}')

# Get the sum of all values after the softmax
sum = tf.reduce_sum(outputs)
print(f'sum of outputs: {sum}')

# Get the index with highest value
prediction = np.argmax(outputs)
print(f'the class with the highest probability: {prediction}')

# building model
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training model
model.fit(training_images, training_labels, epochs=5)
print("Now evaluating...\n\n\n")
# Evaluating model on unseen data
model.evaluate(test_images, test_labels)
