import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.linear_model as lm
import numpy as np
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms
import sklearn.neighbors as n

print("Hello World")
print(pd.__version__)
print(sk.__version__)

df = pd.read_csv("data/house/house.csv")
# model = lm.LinearRegression() # 2 poids
model = n.KNeighborsRegressor(n_neighbors=3)
# model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())
y = df["loyer"]
x = df["surface"].values.reshape(-1, 1)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)
model.fit(xtrain, ytrain)
# print(model.coef_, model.intercept_)
x2 = np.arange(400).reshape(-1, 1)
ypred = model.predict(x2)
print(model.score(xtrain, ytrain))
print(model.score(xtest, ytest))

plt.scatter(df["surface"], df["loyer"], label="Loyer")
plt.plot(x2, ypred, color="red")
plt.title("House")
plt.legend()
plt.show()

