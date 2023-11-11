import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
import matplotlib.pylot as plt 

IMAGE_SIZE = 256
BATCH_SIZE = 32

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    shuffle = True,
    image_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size = BATCH_SIZE
)

print(dataset.shape)