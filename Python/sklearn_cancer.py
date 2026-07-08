import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.neighbors as n
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import numpy as np
import sklearn.model_selection as ms

df = pd.read_csv("data/breast-cancer/data.csv")
y = df["diagnosis"]
x = df.drop(["diagnosis", "id"], axis=1)