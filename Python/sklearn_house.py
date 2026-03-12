import numpy as np
import sklearn
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import pandas as pd
import matplotlib.pyplot as plt
print(sklearn.__version__)

# 1 Pandas DataMart
dataframe = pd.read_csv("data/house/house.csv")

# Déterminer x et y
y = dataframe["loyer"]
x = dataframe["surface"].values.reshape(-1, 1)

# 2 Normalisation : Scaling
# Scaler

# 3 Model
# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())

# 4 Apprentissage supervisé car y est connu
model.fit(x, y)
# print(model.coef_, model.intercept_)

ridge = model.named_steps['ridge']
print(ridge.coef_)

# 5 Score
score = model.score(x, y)
print("Score", score)

# 6 Predict
xnew = np.arange(400).reshape(-1, 1)
ypredicted = model.predict(xnew)

# 7 Dataviz
plt.scatter(dataframe["surface"], dataframe["loyer"])
plt.plot(xnew, ypredicted, color="red")

plt.show()

