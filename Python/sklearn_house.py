import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import numpy as np

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
print(dataframe.head(10))
print(dataframe.describe())

#Dataset
x = dataframe["surface"].values.reshape(-1,1)
y = dataframe["loyer"]

# Instanciation du modèle
model = lm.LinearRegression()
model.fit(x,y)

score = model.score(x,y)
print(score)

xvalues = np.arange(400)
ypred = model.predict(xvalues.reshape(-1,1))

print(model.coef_, model.intercept_)


plt.scatter(dataframe["surface"], dataframe["loyer"])
plt.plot(xvalues, ypred, color="red")
plt.show()
