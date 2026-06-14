# tf_har_to_tflite.py
import tensorflow as tf

model = tf.keras.models.load_model("data/har/har_lstm.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# FP32 + SELECT_TF_OPS pour LSTM
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open("data/har/har_lstm_fp32.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Taille : {len(tflite_model) / 1024:.1f} KB")