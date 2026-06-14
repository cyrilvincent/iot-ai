# h5_to_tflite.py
import numpy as np
import tensorflow as tf

# Charge le modèle H5
model = tf.keras.models.load_model("data/har/har_conv1d.h5")
xtrain = np.load("data/har/raw/xtrain1000.npz")


def representative_dataset():
    for i in range(200):
        sample = xtrain[i:i+1].astype(np.float32)  # shape (1, 100, 3)
        yield [sample]


# Convertisseur
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("data/har/har_conv1d_int8.tflite", "wb") as f:
    f.write(tflite_model)

# Ne fonctionne pas en int8 sur ESP32
# Faut essayer fp32

print(f"Taille modèle : {len(tflite_model) / 1024:.1f} KB")