from PIL import Image
import numpy as np
import struct

path = "data/cifar10/cat.jpg"
img = Image.open(path).resize((32, 32)).convert("RGB")
arr = np.array(img, dtype=np.uint8).flatten()
arr.tofile(path.replace(".jpg", "-32.bin"))

# Code Micropython
# import array
# with open("image.bin", "rb") as f:
#     data = f.read()
# img = array.array("B", data)

# python tflite2tmdl.py ../../python/data/cifar10/cifar10_q.tflite ../../python/data/cifar10/cifar10_q.tmdl int8 1 32,32,3 1
