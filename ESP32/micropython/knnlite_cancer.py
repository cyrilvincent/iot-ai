import knn_lite

# Memory ESP32 Wroom
model = knn_lite.KNNLite(3)
print("Loading CSV")
xtrain = model.load_csv("x_cancer_train.csv")
ytrain = model.load_csv_1d("y_cancer_train.csv")
means = model.load_csv_1d("scaler_cancer_mean.csv")
stds = model.load_csv_1d("scaler_cancer_scale.csv")

x = [[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]]
x = model.scale(x, means, stds)
result = model.train_predict(xtrain, ytrain, x[0], False)
print(result)
print(model.top_distances)
print(model.top_indexes)

x = [[7.76,24.54,47.92,181,0.05263,0.04362,0,0,0.1587,0.05884,0.3857,1.428,2.548,19.15,0.007189,0.00466,0,0,0.02676,0.002783,9.456,30.37,59.16,268.6,0.08996,0.06444,0,0,0.2871,0.07039]]
x = model.scale(x, means, stds)
result = model.train_predict(xtrain, ytrain, x[0], False)
print(result)
print(model.top_distances)
print(model.top_indexes)

