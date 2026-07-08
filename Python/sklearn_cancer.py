import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.neighbors as n
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import numpy as np
import sklearn.model_selection as ms

df = pd.read_csv("data/breast-cancer/data.csv")
y = df["diagnosis"]
x = df.drop(["diagnosis", "id"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

model = n.KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain, ytrain)

print(model.score(xtrain, ytrain))
print(model.score(xtest, ytest))