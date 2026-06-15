import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.neural_network as nn
import sklearn.neighbors as neighbors
import sklearn.ensemble as rf
import sklearn.svm as svm

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)
dataframe = pd.read_csv("data/rul/Battery_RUL.csv")
print(dataframe.head())
print(dataframe.describe())
print(dataframe.shape)

# plt.title('RUL, Remaining Useful Time Histogram')
# sns.histplot(dataframe.RUL, kde=True)
# plt.show()

# correlations = np.abs(dataframe.corr())
# plt.figure(figsize = (15,8))
# sns.heatmap(correlations, annot=True, cbar=False, cmap='Blues', fmt='.1f')
# plt.show()
#
# plt.figure(figsize=(10, 6))
# plt.scatter(dataframe['Max. Voltage Dischar. (V)'], dataframe['RUL'], alpha=0.5)
# plt.title('Scatter Plot: Max. Voltage Discharge vs. Remaining Useful Lifetime')
# plt.xlabel('Max. Voltage Dischar. (V)')
# plt.ylabel('Remaining Useful Lifetime (RUL)')
# plt.grid(True)
# plt.show()
#
# plt.figure(figsize=(10, 6))
# plt.scatter(dataframe['Min. Voltage Charg. (V)'], dataframe['RUL'], alpha=0.5)
# plt.title('Scatter Plot: Min. Voltage Charge vs. Remaining Useful Lifetime')
# plt.xlabel('Min. Voltage Charg. (V)')
# plt.ylabel('Remaining Useful Lifetime (RUL)')
# plt.grid(True)
# plt.show()
#
# plt.figure(figsize=(10, 6))
# plt.scatter(dataframe['Time at 4.15V (s)'], dataframe['RUL'], alpha=0.5)
# plt.title('Scatter Plot: Time at 4.15V vs. Remaining Useful Lifetime')
# plt.xlabel('Time at 4.15V (s)')
# plt.ylabel('Remaining Useful Lifetime (RUL)')
# plt.grid(True)
# plt.show()

y = dataframe['RUL']
x = dataframe.drop(['RUL', 'Cycle_Index'], axis=1)
print(x.head())

np.random.seed(42)
x_train, x_test, y_train, y_test = ms.train_test_split(x, y)

scaler = pp.RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = neighbors.KNeighborsRegressor(n_neighbors=3)
model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)
train_score = model.score(x_train, y_train)
print("Score on Training Set: {:.2%}".format(train_score))
test_score = model.score(x_test, y_test)
print("Score on Test Set: {:.2%}".format(test_score))

model = rf.RandomForestRegressor() # Lent
model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)
train_score = model.score(x_train, y_train)
print("Score on Training Set: {:.2%}".format(train_score))
test_score = model.score(x_test, y_test)
print("Score on Test Set: {:.2%}".format(test_score))


