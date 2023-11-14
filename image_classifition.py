import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.models import Sequential
import pathlib
import os
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataset_url = "https://drive.google.com/file/d/1-SskiVb-mQi-htFOHoeA8nX98TvzudLp/view?usp=sharing"
data_dir = tf.keras.utils.get_file( origin=dataset_url, extract=True,cache_dir="Image-Classifition\data_test")
print(data_dir)
print(type(data_dir))
data_dir = pathlib.Path(data_dir).with_suffix('')
print(data_dir)
print(type(data_dir))
# PATH = "Image-Classifition/data"
# print(PATH)
# print(type(PATH))
data_dir = pathlib.Path(data_dir).with_suffix('')
print(data_dir)
print(type(data_dir))

image_count = len(list(data_dir.glob('*/*.png')))
# roses = list(data_dir.glob('roses/*'))
# im = PIL.Image.open(str(roses[0]))
print(image_count)
batch_size = 32
img_height = 180
img_width = 180
print(data_dir)

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
  )
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# # plt.figure(figsize=(10, 10))
# # for images, labels in train_ds.take(1):
# #   for i in range(9):
# #     ax = plt.subplot(3, 3, i + 1)
# #     plt.imshow(images[i].numpy().astype("uint8"))
# #     plt.title(class_names[labels[i]])
# #     plt.axis("off")

# # for image_batch, labels_batch in train_ds:
# #   print(image_batch.shape)
# #   print(labels_batch.shape)
# #   break

# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# normalization_layer = layers.Rescaling(1./255)

# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
# # Notice the pixel values are now in `[0,1]`.
# # print(np.min(first_image), np.max(first_image))

# num_classes = len(class_names)

# model = Sequential([
#   layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])


# data_augmentation = keras.Sequential(
#   [
#     layers.RandomFlip("horizontal",
#                       input_shape=(img_height,
#                                   img_width,
#                                   3)),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
#   ]
# )

# model = Sequential([
#   data_augmentation,
#   layers.Rescaling(1./255),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Dropout(0.2),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes, name="outputs")
# ])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# # plt.figure(figsize=(10, 10))
# # for images, _ in train_ds.take(1):
# #   for i in range(9):
# #     augmented_images = data_augmentation(images)
# #     ax = plt.subplot(3, 3, i + 1)
# #     plt.imshow(augmented_images[0].numpy().astype("uint8"))
# #     plt.axis("off")

# # model.summary()
# epochs= 15
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )

# # acc = history.history['accuracy']
# # val_acc = history.history['val_accuracy']

# # loss = history.history['loss']
# # val_loss = history.history['val_loss']

# # epochs_range = range(epochs)

# # plt.figure(figsize=(8, 8))
# # plt.subplot(1, 2, 1)
# # plt.plot(epochs_range, acc, label='Training Accuracy')
# # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# # plt.legend(loc='lower right')
# # plt.title('Training and Validation Accuracy')

# # plt.subplot(1, 2, 2)
# # plt.plot(epochs_range, loss, label='Training Loss')
# # plt.plot(epochs_range, val_loss, label='Validation Loss')
# # plt.legend(loc='upper right')
# # plt.title('Training and Validation Loss')


# # Convert the model.
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# # Save the model.
# with open('Image-Classifition-by-Tensorflow\model.tflite', 'wb') as f:
#   f.write(tflite_model)

# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

# img = tf.keras.utils.load_img(
#     sunflower_path, target_size=(img_height, img_width)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch

# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )