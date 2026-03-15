import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
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


# Load the trained model
model = load_model('data/mobilenetv1/mobilenet_model.h5')


# Load and preprocess the image
image_path = 'data/mobilenetv1/dog.jpg'
image = load_and_preprocess_image(image_path)
image_for_prediction = np.expand_dims(image, axis=0)

# Predict the class
predictions = model.predict(image_for_prediction)
predicted_class = (predictions[0][0] > 0.5).astype("int32")


class_names = ['Cat', 'Dog']
predicted_class_name = class_names[predicted_class]

# Display the image with the predicted class
plt.imshow(image)
plt.title(f'Predicted class: {predicted_class_name} @ {abs(predictions[0][0] - 0.5) * 200:.0f}%')
plt.axis('off')
plt.show()

# python resize_img.py quant_img128/ quant_img224/ 224,224,3
# python h5_to_tflite.py ../../python/data/mobilenetv1/mobilenet_model.h5 ../../python/data/mobilenetv1/mobilenet_int8.tflite 1 quant_img224/ 0to1
# python tflite2tmdl.py ../../python/data/mobilenetv1/mobilenet_int8.tflite ../../python/data/mobilenetv1/mobilenet_int8.tmdl int8 1 224,224,3 1
