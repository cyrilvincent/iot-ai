from pathlib import Path

import hls4ml
from tensorflow import keras
import os
import shutil
import platform
import numpy as np

print(hls4ml.__version__)

output_dir = 'data/breast-cancer/hls'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

model = keras.models.load_model("data/breast-cancer/cancer_mlp.h5")
model.summary()

config = hls4ml.utils.config_from_keras_model(model)
# config = hls4ml.utils.config_from_onnx_model()
# config = hls4ml.utils.config_from_pytorch_model()

# Quantisation globale
# config['Model']['Precision'] = 'fixed<8,2>' # 16, 6 par défaut

# Quantisation par layer
# config['LayerName']['dense']['Precision']['weight']     = 'fixed<8,2>'
# config['LayerName']['dense']['Precision']['bias']       = 'fixed<8,2>'
# config['LayerName']['dense']['Precision']['result']     = 'fixed<16,6>'
# config['LayerName']['dense_1']['Precision']['weight']   = 'fixed<8,2>'
# config['LayerName']['dense_2']['Precision']['weight']   = 'fixed<8,2>'

print(config)
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir=output_dir,
    backend='Vivado',  # Vivado VivadoAccelerator Quartus Catapult
    board='zcu102',
)

hls_model.write()  # Création du C++

if platform.system() == "Linux":
    hls_model.compile()  # Création des bitmaps et des metrics
    x = np.array([[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]])
    result = hls_model.predict(x)
    print(result)



