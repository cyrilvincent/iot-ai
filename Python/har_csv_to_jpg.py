import numpy as np
import pandas as pd
from PIL import Image


df = pd.read_csv("data/har/Accelerometer.csv")
df["time"] = (df["seconds_elapsed"] * 1000).astype(np.int16)
# df.x = (df.x * 2 ** 15).astype(np.int16)
# df.y = (df.y * 2 ** 15).astype(np.int16)
# df.z = (df.z * 2 ** 15).astype(np.int16)
df = df.drop(["time", "seconds_elapsed"], axis=1)
df = df[["x", "y", "z"]]

df = (df - df.min().min()) / df.max().max()
df = np.clip(df * 255, 0, 255)
df = df.astype(np.uint8)

print(df)

hop_length=128
num_file = 0
for batch in range(int(len(df) / hop_length)):
    hop_df = df[batch * hop_length: batch * hop_length + hop_length]
    hop = hop_df.values.reshape(16, 24)
    im = Image.fromarray(hop).convert("RGB")
    im.save(f"data/har/quant/har-{num_file:02d}.bmp")
    num_file += 1






