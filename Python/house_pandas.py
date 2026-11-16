import pandas as pd
import matplotlib.pyplot as plt

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
print(dataframe.head(10))
print(dataframe.describe())

dataframe.hist(bins=20)
plt.show()

plt.scatter(dataframe["surface"], dataframe["loyer"])
plt.show()

