import pandas as pd
import sklearn.model_selection as ms
import sklearn.ensemble as rf
import sklearn.preprocessing as pp
import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree as tree
import emlearn

df = pd.read_csv('data/heart/data_cleaned_up.csv')
df["rnd"] = np.random.rand(len(df))
x = df.drop('num', axis=1)
y = df['num'].values

x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.2, random_state=42)

scaler = pp.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape)

#model = n.KNeighborsClassifier(n_neighbors=3)
model = rf.RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

print(model.feature_importances_)

plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()

np.savetxt('data/heart/scaler_mean.csv',  scaler.mean_,  delimiter=',')
np.savetxt('data/heart/scaler_scale.csv', scaler.scale_, delimiter=',')

cmodel = emlearn.convert(model, method='inline')
cmodel.save(file="data/heart/heart_rf_model.csv", name='rf', format='csv')
cmodel.save(file='data/heart/heart_rf_model.h', name='breast_cancer_rf')

tree.export_graphviz(model.estimators_[0], out_file='data/heart/tree.dot', feature_names=x.columns, class_names=["0", "1"])
