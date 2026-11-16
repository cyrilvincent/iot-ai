import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.linear_model as lm
import numpy as np
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
print(dataframe.head(10))
print(dataframe.describe())

#Dataset
x = dataframe["surface"].values.reshape(-1,1)
y = dataframe["loyer"]

xtrain, xtest, ytrain, ytest = ms.train_test_split(x,y,train_size=0.8, test_size=0.2, random_state=42)

# Instanciation du modèle
# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(4), lm.Ridge())
model.fit(xtrain, ytrain)

train_score = model.score(xtrain,ytrain)
print(train_score)
test_score = model.score(xtest,ytest)
print(test_score)

xvalues = np.arange(400)
ypred = model.predict(xvalues.reshape(-1,1))

# print(model.coef_, model.intercept_)


plt.scatter(dataframe["surface"], dataframe["loyer"])
plt.plot(xvalues, ypred, color="red")
plt.show()
