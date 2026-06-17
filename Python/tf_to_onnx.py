import tf2onnx
import tensorflow as tf

# Charge le .h5
model = tf.keras.models.load_model("data/breast-cancer/cancer_mlp.h5")

# Convertit en ONNX
input_signature = [tf.TensorSpec([1, 30], tf.float32, name="input")]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)

import onnx
onnx.save(onnx_model, "data/breast-cancer/cancer_mlp.h5.onnx")
