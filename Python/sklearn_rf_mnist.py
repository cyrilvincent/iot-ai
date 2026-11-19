import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.ensemble as rf
import sklearn.preprocessing as pp
import matplotlib.pyplot as plt

np.random.seed(42)

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    xtrain, ytrain = f["x_train"], f["y_train"]
    xtest, ytest = f["x_test"], f["y_test"]

xtrain = xtrain.reshape(-1, 28*28).astype(np.float64)  # 784
xtest = xtest.reshape(-1, 28*28).astype(np.float64)

nb = 60000
xtrain = xtrain[:nb]
xtest = xtest[:nb]
ytrain = ytrain[:nb]
ytest = ytest[:nb]

xtrain = xtrain / 255.
xtest = xtest / 255.



# Inutilisable sur ESP32 trp de features
model = rf.RandomForestClassifier(n_estimators=100, random_state=42) # Ne marche pas au dessus
model.fit(xtrain, ytrain)
score_train = model.score(xtrain, ytrain)
print(score_train)

plt.matshow(model.feature_importances_.reshape(28, 28))
plt.show()



# cmodel = emlearn.convert(model, method='inline')
# cmodel.save(file="data/mnist/mnist_rf_model.csv", name='rf', format='csv')

print("N_TREES :", model.n_estimators)
print("N_NODES :", sum(tree.tree_.node_count for tree in model.estimators_))
print("N_CLASSES:", model.n_classes_)
print("N_FEATURES:", model.n_features_in_)




