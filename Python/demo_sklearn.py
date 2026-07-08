import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import numpy as np

print(sklearn.__version__)

df = pd.read_csv("data/house/house.csv")
plt.scatter(df["surface"], df["loyer"])


y = df["loyer"]
x = df["surface"].values.reshape(-1, 1)

# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge()) # est identique à LinearRegression

model.fit(x, y)

print(model.score(x, y))
print(model[-1].coef_)

x = np.arange(400).reshape(-1, 1)
ypred = model.predict(x)

# print(f"Slope: {model.coef_}, Intercept: {model.intercept_}")

plt.plot(x, ypred, color="red")
plt.show()



