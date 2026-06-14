import pandas as pd

df = pd.read_csv("Accelerometer.csv")
df = df[["x", "y", "z"]]
df.to_csv("accelerometer_rnn.csv", index=False)
