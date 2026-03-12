import knn_lite

model = knn_lite.KNNLite(3)
print("Loading CSV")
xtrain = model.load_csv("data/mnist/x_mnist_train.csv")
ytrain = model.load_csv_1d("data/mnist/y_mnist_train.csv")
for i in range(10):
    x = model.load_csv_1d(f"data/mnist/{i}.csv")
    print(f"Try to predict {i}")
    result = model.train_predict(xtrain, ytrain, x, False)
    print(i, result, model.top_indexes)

