import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.neural_network as nn
import sklearn.preprocessing as pp
import emlearn

np.random.seed(42)

pd.set_option('display.max_columns', None)
dataframe = pd.read_csv("data/heart/data_cleaned_up.csv")

y = dataframe["num"]
x = dataframe.drop(["num"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2,)

scaler = pp.StandardScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

model = nn.MLPClassifier(hidden_layer_sizes=(8,), random_state=1, max_iter=1000)
model.fit(xtrain, ytrain)
score_train = model.score(xtrain, ytrain)
score_test = model.score(xtest, ytest)
print(score_train, score_test)

for i in range(len(model.coefs_)):
    nb_input = xtrain.shape[1] if i == 0 else model.hidden_layer_sizes[i - 1]
    nb_output = model.n_outputs_ if i == len(model.coefs_) - 1 else model.hidden_layer_sizes[i]
    print(f"Layer {i} {nb_input}x{nb_output}")
    print(model.coefs_[i].T)
