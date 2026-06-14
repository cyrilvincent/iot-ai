# https://github.com/ani8897/Human-Activity-Recognition
import keras.utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalAveragePooling1D, Dropout

np.random.seed(42)

print('Loading data ...')
pd.set_option('display.max_columns', None)
data = pd.read_csv('data/har/raw/Phones_accelerometer.csv')
data = data[data["Model"] == "nexus4"]
data = data.drop(labels=['Arrival_Time', 'Creation_Time', 'Index', "Model", "Device"], axis=1)
# User est gardé pour le split
data = data.dropna()

print(data['gt'].unique())
le = pp.LabelEncoder()
data['gt'] = le.fit_transform(data['gt'])

# Split par utilisateur
users = data['User'].unique()
print(f"Utilisateurs : {users}")
train_users = users[:round(0.8 * len(users))]
train_data = data[data['User'].isin(train_users)].drop('User', axis=1)
test_data  = data[~data['User'].isin(train_users)].drop('User', axis=1)

ytrain = train_data[['gt']]
ytest  = test_data[['gt']]
xtrain = train_data.drop(['gt'], axis=1)
xtest  = test_data.drop(['gt'], axis=1)

scaler = pp.StandardScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest  = scaler.transform(xtest)

ytrain = keras.utils.to_categorical(ytrain)
ytest  = keras.utils.to_categorical(ytest)

time_steps = 100
stride = 10

def create_sequences(X, y, window_size, step):
    Xs, ys = [], []
    for i in range(0, len(X) - window_size, step):
        Xs.append(X[i:i + window_size])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)

xtrain, ytrain = create_sequences(xtrain, ytrain, time_steps, stride)
xtest,  ytest  = create_sequences(xtest,  ytest,  time_steps, stride)

print(f"xtrain : {xtrain.shape}, xtest : {xtest.shape}")

np.save("data/har/xtrain.npy", xtrain)  # Pour la calibration INT8

model = Sequential()
model.add(Conv1D(64, kernel_size=5, activation='relu', input_shape=(time_steps, xtrain.shape[2])))
model.add(Dropout(0.3))
model.add(Conv1D(32, kernel_size=3, activation='relu'))
model.add(Dropout(0.3))
model.add(GlobalAveragePooling1D())
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

batch_size = 128
model.fit(xtrain, ytrain, epochs=5, batch_size=batch_size,
          validation_data=(xtest, ytest), shuffle=True)

scores = model.evaluate(xtest, ytest, batch_size=1024)
print(f"Accuracy: {scores[1]*100:.2f}%")

model.save("data/har/har_conv1d.h5")
