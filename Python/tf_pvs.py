# https://www.kaggle.com/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets
# https://www.kaggle.com/code/outathyamohanta/feature-engineering-and-model-training

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import pickle
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM, BatchNormalization
import keras.utils

np.random.seed(42)

print('Loading data ...')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

dir = "data/pvs"

# segments = []
# for i in range(1, 10):
#     subdir = f"{dir}/PVS {i}"
#     print(f"Scan directory {subdir}")
#     path = f"{subdir}/dataset_labels.csv"
#     print(f"Loading {path}")
#     label_df = pd.read_csv(path)
#     print(label_df.shape)
#     path = f"{subdir}/dataset_mpu_left.csv"
#     print(f"Loading {path}")
#     mpu_df = pd.read_csv(path)
#     print(mpu_df.shape)
#     left_df = mpu_df.join(label_df)
#     left_df["pvs"] = i
#     segments.append(left_df)
#     print(f"left_df shape {left_df.shape}")
#     path = f"{subdir}/dataset_mpu_right.csv"
#     print(f"Loading {path}")
#     mpu_df = pd.read_csv(path)
#     print(mpu_df.shape)
#     right_df = mpu_df.join(label_df)
#     print(f"right_df shape {right_df.shape}")
#     right_df["pvs"] = i
#     segments.append(right_df)
#
# print("Concat")
# df = pd.concat(segments, ignore_index=True)
# print(f"df shape: {df.shape}")
#
# test_data = df[df["pvs"].isin([5, 6])]
# train_data = df[~df["pvs"].isin([5, 6])]
# print(train_data.shape)
# print(test_data.shape)
# with open(f"{dir}/pvs_train.pkl", "wb") as f:
#     pickle.dump(train_data, f)
# with open(f"{dir}/pvs_test.pkl", "wb") as f:
#     pickle.dump(test_data, f)
#


with open(f"{dir}/pvs_train.pkl", "rb") as f:
    train_data = pickle.load(f)
with open(f"{dir}/pvs_test.pkl", "rb") as f:
    test_data = pickle.load(f)
print(train_data.shape)
print(test_data.shape)


def create_windows(df, window_size=100, step_size=50):
    print(f"Creating windows {100}x{step_size} for {len(df)} rows")
    x, y = [], []
    feature_cols = ['acc_x_dashboard', 'acc_y_dashboard', 'acc_z_dashboard']
    for i in range(0, len(df) - window_size, step_size):
        window = df.iloc[i: i + window_size]
        x.append(window[feature_cols].values)
        y.append(window['paved_road'].mode()[0])  # Predict 'Paved' vs 'Unpaved/Rough'
    return np.array(x), np.array(y)


window_size = 100
step_size = 50
xtrain, ytrain = create_windows(train_data, window_size, step_size)
xtest, ytest = create_windows(test_data, window_size, step_size)
print(f"xtrain shape: {xtrain.shape}")
print(f"ytrain shape: {ytrain.shape}")
print(f"xtest shape: {xtest.shape}")
print(f"ytest shape: {ytest.shape}")

scaler = pp.StandardScaler()
xtrain_reshaped = xtrain.reshape(-1, xtrain.shape[-1])
scaler.fit(xtrain_reshaped)
xtrain_scaled = scaler.transform(xtrain_reshaped).reshape(xtrain.shape)
xtest_reshaped = xtest.reshape(-1, xtest.shape[-1])
xtest_scaled = scaler.transform(xtest_reshaped).reshape(xtest.shape)

input_shape = (xtrain.shape[1], xtrain.shape[2])
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    xtrain_scaled, ytrain,
    epochs=15,
    batch_size=32,
    validation_data=(xtest_scaled, ytest)
)

model.save("data/pvs/pvs.h5")

ypred = (model.predict(xtest_scaled) > 0.5).astype("int32")

samples_to_show = 350
actual = ytest[:samples_to_show]
predicted = ypred[:samples_to_show].flatten()
plt.figure(figsize=(15, 5))

plt.step(range(samples_to_show), actual, label='Actual (Fiat Palio)', alpha=0.7, linewidth=2, where='post')
plt.step(range(samples_to_show), predicted, label='AI Predicted', linestyle='--', alpha=0.9, linewidth=2, color='red', where='post')
plt.title('Real-Time Generalization: Actual vs. Predicted Labels (Fiat Palio)', fontsize=14)
plt.xlabel('Time Windows (Sequence)', fontsize=12)
plt.ylabel('Road Type (0: Paved | 1: Unpaved)', fontsize=12)
plt.yticks([0, 1], ['Paved', 'Unpaved'])
plt.legend(loc='upper right')
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()



