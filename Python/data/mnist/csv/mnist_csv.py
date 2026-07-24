import numpy as np
l = []
with open("0.csv") as f:
    for row in f:
        l.append(float(row))
array = np.array(l)
array /= 255
array = array.reshape(28, 28)
print(array)