# h5_to_tflite.py
import numpy as np
import tensorflow as tf

# Charge le modèle H5
h5 = "data/har/har_conv1d.h5"
h5 = "data/har/har_rnn.h5"
h5 = "data/har/har_lstm.h5"
model = tf.keras.models.load_model(h5)
xtrain = np.load("data/har/raw/xtrain1000.npy")


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

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open(h5.replace(".h5", "_int8.tflite"), "wb") as f:
    f.write(tflite_model)

print(f"Taille modèle : {len(tflite_model) / 1024:.1f} KB")