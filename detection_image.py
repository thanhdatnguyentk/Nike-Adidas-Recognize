import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TF_MODEL_FILE_PATH = 'Image-Classifition-by-Tensorflow\model.tflite' # The default path to the saved TensorFlow Lite model

interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)

# ------------------ check load file ---------------
classify_lite = interpreter.get_signature_runner('serving_default')
# print(classify_lite)

batch_size = 32
img_height = 180
img_width = 180


class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url, cache_dir='data')

sunflower_path = "Image-Classifition-by-Tensorflow/tải xuống.jpg"
img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch


predictions_lite = classify_lite(sequential_1_input=img_array)['outputs']
score_lite = tf.nn.softmax(predictions_lite)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)
