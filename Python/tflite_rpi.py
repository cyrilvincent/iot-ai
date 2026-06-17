import numpy as np
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter(model_path="data/h5/cholletmodel-mnist.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_test = f['x_test']

x_test = x_test / 255.
input_data = x_test[0].astype(np.float32).reshape(1,28,28,1)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)