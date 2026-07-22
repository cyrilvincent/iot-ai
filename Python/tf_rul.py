import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import tensorflow as tf
import pandas
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

dataframe = pandas.read_csv("data/rul/Battery_RUL.csv")
y = dataframe["RUL"]
x = dataframe.drop("RUL", axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.MinMaxScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)
yscaler = pp.MinMaxScaler()
yscaler.fit(ytrain.values.reshape(-1, 1))
ytrain = yscaler.transform(ytrain.values.reshape(-1, 1))
ytest = yscaler.transform(ytest.values.reshape(-1, 1))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation="relu", input_shape=(x.shape[1],)),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
  ])

model.compile(loss="mse", optimizer="rmsprop")
model.summary()

model.fit(xtrain, ytrain, epochs=10)

ypred = model.predict(xtest)
ypred = yscaler.inverse_transform(ypred)
print(ypred)
