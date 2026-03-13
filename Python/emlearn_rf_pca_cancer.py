import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.ensemble as rf
import sklearn.preprocessing as pp
import emlearn
import pca_lite

np.random.seed(42)

pd.set_option('display.max_columns', None)
dataframe = pd.read_csv("data/breast-cancer/data.csv")

y = dataframe["diagnosis"]
x = dataframe.drop(["diagnosis", "id"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2,)

scaler = pp.StandardScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

pca = pca_lite.PCALite(10, 5)  # Réduction de 30 à 12 dimensions
xtrain = pca.reduce2(xtrain)
xtest = pca.reduce2(xtest)

model = rf.RandomForestClassifier(n_estimators=10, max_depth=10, max_features=2, random_state=1)
model.fit(xtrain, ytrain)
score_train = model.score(xtrain, ytrain)
score_test = model.score(xtest, ytest)
print(score_train, score_test)

cmodel = emlearn.convert(model, method='inline')
cmodel.save(file="data/breast-cancer/cancer_pca_rf_model.csv", name='rf', format='csv')

np.savetxt('data/breast-cancer/scaler_pca_cancer_mean.csv',  scaler.mean_,  delimiter=',')
np.savetxt('data/breast-cancer/scaler_pca_cancer_scale.csv', scaler.scale_, delimiter=',')

# Il serait possible de faire un TOP 10 + 1 PCA L1 et L2 pour les autres colonnes ce qui ferais 12 features
