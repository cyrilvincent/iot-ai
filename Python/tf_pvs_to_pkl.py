# https://www.kaggle.com/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import pickle


np.random.seed(42)

print('Loading data ...')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

dir = "data/pvs"

segments = []
for i in range(1, 10):
    subdir = f"{dir}/PVS {i}"
    print(f"Scan directory {subdir}")
    path = f"{subdir}/dataset_labels.csv"
    print(f"Loading {path}")
    label_df = pd.read_csv(path)
    print(label_df.shape)
    path = f"{subdir}/dataset_mpu_left.csv"
    print(f"Loading {path}")
    mpu_df = pd.read_csv(path)
    print(mpu_df.shape)
    left_df = mpu_df.join(label_df)
    left_df["pvs"] = i
    segments.append(left_df)
    print(f"left_df shape {left_df.shape}")
    path = f"{subdir}/dataset_mpu_right.csv"
    print(f"Loading {path}")
    mpu_df = pd.read_csv(path)
    print(mpu_df.shape)
    right_df = mpu_df.join(label_df)
    print(f"right_df shape {right_df.shape}")
    right_df["pvs"] = i
    segments.append(right_df)

print("Concat")
df = pd.concat(segments, ignore_index=True)
print(f"df shape: {df.shape}")

test_data = df[df["pvs"].isin([5, 6])]
train_data = df[~df["pvs"].isin([5, 6])]
print(train_data.shape)
print(test_data.shape)
with open(f"{dir}/pvs_train.pkl", "wb") as f:
    pickle.dump(train_data, f)
with open(f"{dir}/pvs_test.pkl", "wb") as f:
    pickle.dump(test_data, f)

