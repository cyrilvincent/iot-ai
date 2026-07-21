import numpy as np
import sklearn.neural_network as nn

np.random.seed(42)

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    xtrain, ytrain = f["x_train"], f["y_train"]
    xtest, ytest = f["x_test"], f["y_test"]

print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
xtrain = xtrain.reshape(-1, 28*28)
xtest = xtest.reshape(-1, 28*28)

model = nn.MLPClassifier(hidden_layer_sizes=(500,200,100))
model.fit(xtrain, ytrain)
ypredicted = model.predict(xtest)
score = model.score(xtest, ytest)
print(f"Score: {score:.3f}")



