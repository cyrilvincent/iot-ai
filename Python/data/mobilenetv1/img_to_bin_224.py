from PIL import Image
import numpy as np
import struct

path = "data/mobilenetv1/dog.jpg"
img = Image.open(path).resize((224, 224)).convert("RGB")
arr = np.array(img, dtype=np.uint8).flatten()
arr.tofile(path.replace(".jpg", "-224.bin"))

# Code Micropython
# import array
# with open("image.bin", "rb") as f:
#     data = f.read()
# img = array.array("B", data)
