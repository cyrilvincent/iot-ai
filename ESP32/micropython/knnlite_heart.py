import knn_lite

model = knn_lite.KNNLite(3)
xtrain = model.load_csv("x_train.csv")
ytrain = model.load_csv_1d("y_train.csv")
means = model.load_csv_1d("scaler_mean.csv")
stds = model.load_csv_1d("scaler_scale.csv")
xnorm = model.scale(xtrain, means, stds)

x = [[28, 1, 2, 130, 132, 0, 2, 185, 0, 0]]
x = model.scale(x, means, stds)
result = model.train_predict(xnorm, ytrain, x[0])
print(result)

x = [[65, 1, 4, 130, 275, 0, 1, 115, 1, 1]]
x = model.scale(x, means, stds)
result = model.train_predict(xnorm, ytrain, x[0])
print(result)




