import knn_lite

model = knn_lite.KNNLite(3)
print("Loading CSV")
xtrain = model.load_csv("x_mnist_train.csv")
ytrain = model.load_csv_1d("y_mnist_train.csv")
for i in range(10):
    x = model.load_csv_1d(f"data/mnist/{i}.csv")
    print(f"Predict {i}")
    result = model.train_predict(xtrain, ytrain, x, True)
    print(i, result)

