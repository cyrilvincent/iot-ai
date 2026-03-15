from PIL import Image
import numpy as np
import struct

path = "cat.jpg"
img = Image.open(path).resize((96, 96)).convert("RGB")
arr = np.array(img, dtype=np.uint8).flatten()
arr.tofile(path.replace(".jpg", "-96.bin"))

# Code Micropython
# import array
# with open("image.bin", "rb") as f:
#     data = f.read()
# img = array.array("B", data)

# python h5_to_tflite.py ../../python/data/imagenet/mnist_cnn_mlp.h5 ../../python/data/imagenet/mnist_cnn_mlp_int8.tflite 1 quant_img_mnist/ 0to1
# python tflite2tmdl.py ../../python/data/imagenet/mbnet96_0.25_q.tflite ../../python/data/imagenet/mbnet96_q.tmdl int8 1 96,96,3 1
