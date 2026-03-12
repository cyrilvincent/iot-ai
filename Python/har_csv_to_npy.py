import numpy as np
import pandas as pd
import array
from windower import TriaxialWindower, empty_array
import timebased

hop_length = 64
window_length = 128
file_length = 60

df = pd.read_csv("data/har/Accelerometer.csv")
df["time"] = (df["seconds_elapsed"] * 1000).astype(np.int16)
df.x = (df.x * 2 ** 15).astype(np.int16)
df.y = (df.y * 2 ** 15).astype(np.int16)
df.z = (df.z * 2 ** 15).astype(np.int16)
df = df.drop(["time", "seconds_elapsed"], axis=1)
df = df[["x", "y", "z"]]
print(df.values)

x_values = empty_array('h', hop_length)
y_values = empty_array('h', hop_length)
z_values = empty_array('h', hop_length)
windower = TriaxialWindower(window_length)
x_window = empty_array('h', window_length)
y_window = empty_array('h', window_length)
z_window = empty_array('h', window_length)
n_features = timebased.N_FEATURES
# features_typecode = timebased.DATA_TYPECODE
# features = array.array(features_typecode, (0 for _ in range(n_features)))
results = []


def copy_array_into(source, target):
    assert len(source) == len(target)
    for i in range(len(target)):
        target[i] = source[i]


num_file = 0
for batch in range(int(len(df) / hop_length)):
    xvalues = df.values[batch * hop_length: batch * hop_length + hop_length]
    yvalues = df.values[batch * hop_length: batch * hop_length + hop_length]
    zvalues = df.values[batch * hop_length: batch * hop_length + hop_length]
    for i in range(hop_length):
        xs = array.array("h", xvalues[i])
        ys = array.array("h", yvalues[i])
        zs = array.array("h", zvalues[i])
        windower.push(xs, ys, zs)
        if windower.full():
            copy_array_into(windower.x_values, x_window)
            copy_array_into(windower.y_values, y_window)
            copy_array_into(windower.z_values, z_window)
            features_typecode = timebased.DATA_TYPECODE
            ff = timebased.calculate_features_xyz((x_window, y_window, z_window))
            results.append(ff)
            if len(results) == file_length:
                results = np.array(results, dtype=np.int16)
                print(results.shape)
                np.save(f"data/har/har-{num_file:02d}.npy", results)
                results = []
                num_file += 1




