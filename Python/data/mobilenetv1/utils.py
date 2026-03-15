import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def preprocess(image, label):
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label

def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))  # Load the image and resize it to 224x224
    image = img_to_array(image) / 255.0  # Normalize the image to [0, 1]
    return image
