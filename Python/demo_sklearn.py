import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm

print(sklearn.__version__)

df = pd.read_csv("data/house/house.csv")
plt.scatter(df["surface"], df["loyer"])


y = df["loyer"]
x = df["surface"].values.reshape(-1, 1)

model = lm.LinearRegression()
model.fit(x, y)

print(model.score(x, y))

ypred = model.predict(x)

print(f"Slope: {model.coef_}, Intercept: {model.intercept_}")

plt.plot(x, ypred, color="red")
plt.show()



