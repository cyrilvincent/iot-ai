import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import numpy as np
import sklearn.model_selection as ms

print(sklearn.__version__)

df = pd.read_csv("data/house/house.csv")
plt.scatter(df["surface"], df["loyer"])


y = df["loyer"]
x = df["surface"].values.reshape(-1, 1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge()) # est identique à LinearRegression

model.fit(xtrain, ytrain)

print(model.score(xtrain, ytrain))
print(model.score(xtest, ytest))
print(model[-1].coef_)

x = np.arange(400).reshape(-1, 1)
ypred = model.predict(x)

# print(f"Slope: {model.coef_}, Intercept: {model.intercept_}")

plt.plot(x, ypred, color="red")
plt.show()



