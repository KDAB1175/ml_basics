import tensorflow as tf
import matplotlib.pyplot as plt
print(tf.__version__)


mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

plt.imshow(training_images[0])

print(training_labels[0])
print(training_images[0])