import tensorflow.keras as keras
import numpy as np

model = keras.models.load_model("data/heart/heart_mlp.h5")
x = np.array([[28,1,2,130,132,0,2,185,0,0]])
#Manque la normalisation
y = model.predict(x)
print(y)