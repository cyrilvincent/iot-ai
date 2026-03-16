import sklearn.neighbors as n
import numpy as np
import matplotlib.pyplot as plt

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    xtrain, ytrain = f["x_train"], f["y_train"]
    xtest, ytest = f["x_test"], f["y_test"]

sample = 1000
xtrain = xtrain[::sample]
ytrain = ytrain[::sample]

xtrain = xtrain.reshape(-1, 28*28)  # 784
xtest = xtest.reshape(-1, 28*28)

model = n.KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

np.savetxt('data/mnist/x_mnist_train.csv', xtrain, delimiter=',', fmt='%.0f')
np.savetxt('data/mnist/y_mnist_train.csv', ytrain,   delimiter=',', fmt='%d')

# for x, y in zip(xtest[:100], ytest[:100]):
#     np.savetxt(f"data/mnist/{y}.csv", x, delimiter=',', fmt='%.0f')

ypredicted = model.predict(xtest)

xtest = xtest.reshape(-1, 28, 28)
select = np.random.randint(xtest.shape[0], size=12)

for index, value in enumerate(select):
    plt.subplot(3, 4, index + 1)
    plt.axis("off")
    plt.imshow(xtest[value], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title(f"Predicted {ypredicted[value]}")
plt.show()

errors = ytest != ypredicted
xerrors = xtest[errors]
yerrors = ypredicted[errors]

select = np.random.randint(xerrors.shape[0], size=12)

for index, value in enumerate(select):
    plt.subplot(3, 4, index + 1)
    plt.axis("off")
    plt.imshow(xerrors[value], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title(f"Predicted {yerrors[value]}")
plt.show()



