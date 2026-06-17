import numpy as np
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter(model_path="data/breast-cancer/cancer_mlp.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

x = np.array([[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]])

means = np.array([6.981000, 9.710000, 43.790000, 143.500000, 0.052630, 0.019380, 0.000000, 0.000000, 0.106000, 0.049960, 0.111500, 0.360200, 0.757000, 6.802000, 0.001713, 0.002252, 0.000000, 0.000000, 0.007882, 0.000895, 7.930000, 12.020000, 50.410000, 185.200000, 0.071170, 0.027290, 0.000000, 0.000000, 0.156500, 0.055040])
stds = np.array([28.110000, 39.280000, 188.500000, 2501.000000, 0.163400, 0.345400, 0.426800, 0.201200, 0.304000, 0.097440, 2.873000, 4.885000, 21.980000, 542.200000, 0.031130, 0.135400, 0.396000, 0.052790, 0.078950, 0.029840, 36.040000, 49.540000, 251.200000, 4254.000000, 0.222600, 1.058000, 1.252000, 0.291000, 0.663800, 0.207500])
x = (x - means) / stds

input_data = x[0].astype(np.float32).reshape(1,30)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
print(np.argmax(output_data))
