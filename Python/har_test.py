import numpy as np

data = np.load("har_trees/uci_har.testdata.npz")
x = data["X"]
print(x.shape)
print(x)

data = np.load("data/har/har-10.npy")
print(data.shape)
print(data)