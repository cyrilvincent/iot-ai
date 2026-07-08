import pandas as pd

df = pd.read_csv("data/heart/data_cleaned_up.csv")

ok = df[df["num"] == 0]
ko = df[df["num"] == 1]

print(ok["sex"].describe())
print(ko["sex"].describe())
print(ok["chol"].describe())
print(ko["chol"].describe())
print(ok["thalach"].describe())
print(ko["thalach"].describe())