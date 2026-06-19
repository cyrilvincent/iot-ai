# https://www.kaggle.com/datasets/outofskills/driving-behavior/data
import keras.utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalAveragePooling1D, Dropout, LSTM

np.random.seed(42)

print('Loading data ...')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
train_data = pd.read_csv("data/driving_behavior/train_motion_data.csv", index_col="Timestamp")
test_data = pd.read_csv("data/driving_behavior/test_motion_data.csv", index_col="Timestamp")
print(train_data.describe())

le = pp.LabelEncoder()
le.fit(train_data["Class"])
train_data["Class_encoded"] = le.transform(train_data["Class"])
test_data["Class_encoded"] = le.transform(test_data["Class"])
print(train_data["Class"].unique(), train_data["Class_encoded"].unique())

xtrain = train_data[["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]]
ytrain = train_data["Class_encoded"]
xtest = test_data[["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]]
ytest = test_data["Class_encoded"]

scaler = pp.StandardScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

ytrain = keras.utils.to_categorical(ytrain)
ytest = keras.utils.to_categorical(ytest)

time_steps = 40
stride = 1


def create_sequences(x, y, window_size, step):
    xs, ys = [], []
    for i in range(0, len(x) - window_size, step):
        xs.append(x[i:i + window_size])
        ys.append(y[i + window_size])
    return np.array(xs), np.array(ys)


xtrain, ytrain = create_sequences(xtrain, ytrain, time_steps, stride)
xtest, ytest = create_sequences(xtest, ytest, time_steps, stride)

print(f"xtrain : {xtrain.shape}, xtest : {xtest.shape}")

# Trop peu de données pour que celà fonctionnes
model = Sequential()
model.add(LSTM(32, input_shape=(time_steps, xtrain.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(16))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

batch_size = 16
model.fit(xtrain, ytrain, epochs=20, batch_size=batch_size, shuffle=True, validation_data=(xtest, ytest))

scores = model.evaluate(xtest, ytest, batch_size=1024)
print(f"Accuracy: {scores[1]*100:.2f}%")
#
# model.save("data/har/har_lstm.h5")
