import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import load_and_preprocess_image

# Load the trained model
model = load_model('mobilenet_model.h5')

# Load and preprocess the image
image_path = 'cat.jpg'
image = load_and_preprocess_image(image_path)
image_for_prediction = np.expand_dims(image, axis=0)  # Add batch dimension

# Predict the class
predictions = model.predict(image_for_prediction)
predicted_class = (predictions[0][0] > 0.5).astype("int32")

class_names = ['Cat', 'Dog']
predicted_class_name = class_names[predicted_class]

# Display the image with the predicted class
plt.imshow(image)
plt.title(f'Predicted class: {predicted_class_name}')
plt.axis('off')
plt.show()
