import onnx
from onnx2keras import onnx_to_keras
import keras

print(keras.__version__)
name = "data/breast-cancer/cancer_mlp.onnx"
onnx_model = onnx.load(name)
keras_model = onnx_to_keras(onnx_model, ['input'])  # Incompatibilité keras 3
keras_model.summary()
keras_model.save(name + ".h5")
