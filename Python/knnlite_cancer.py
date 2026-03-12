import knn_lite

model = knn_lite.KNNLite(3)
print("Loading CSV")
xtrain = model.load_csv("data/breast-cancer/x_cancer_train.csv")
ytrain = model.load_csv_1d("data/breast-cancer/y_cancer_train.csv")
means = model.load_csv_1d("data/breast-cancer/scaler_cancer_mean.csv")
stds = model.load_csv_1d("data/breast-cancer/scaler_cancer_scale.csv")
xnorm = model.scale(xtrain, means, stds)

# x = [[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]]
# x = model.scale(x, means, stds)
# result = model.train_predict(xnorm, ytrain, x[0], False)
# print(result)
# print(model.top_distances)
# print(model.top_indexes)

x = [[13,21.82,87.5,519.8,0.1273,0.1932,0.1859,0.09353,0.235,0.07389,0.3063,1.002,2.406,24.32,0.005731,0.03502,0.03553,0.01226,0.02143,0.003749,15.49,30.73,106.2,739.3,0.1703,0.5401,0.539,0.206,0.4378,0.1072]]
x = model.scale(x, means, stds)
result = model.train_predict(xnorm, ytrain, x[0], False)
print(result)
print(model.top_distances)
print(model.top_indexes)

