import pandas as pd
import sklearn.model_selection as ms
import sklearn.neighbors as n
import numpy as np
import sklearn.preprocessing as pp

df = pd.read_csv('data/breast-cancer/data.csv')
x = df.drop(["diagnosis", "id"], axis=1).values
y = df['diagnosis'].values

x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.2, random_state=42)

scaler = pp.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape)

model = n.KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

np.savetxt('data/breast-cancer/scaler_cancer_mean.csv',  scaler.mean_,  delimiter=',')
np.savetxt('data/breast-cancer/scaler_cancer_scale.csv', scaler.scale_, delimiter=',')
np.savetxt('data/breast-cancer/x_cancer_train.csv', x_train, delimiter=',', fmt='%.6f')
np.savetxt('data/breast-cancer/y_cancer_train.csv', y_train,   delimiter=',', fmt='%d')
