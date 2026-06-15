import torch
import sklearn.preprocessing as pp
import pandas as pd
# noinspection PyUnresolvedReference
from executorch.extension.pybindings.portable_lib import _load_for_executorch

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)

scaler = pp.MinMaxScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

module = _load_for_executorch("data/breast-cancer/cancer_mlp_int8.pte")

correct = 0
total = len(x_scaled)

le = pp.LabelEncoder()  # 0 et 1 sont en string
y_encoded = le.fit_transform(y)

for i in range(total):
    sample = torch.tensor(x_scaled[i:i+1], dtype=torch.float32)
    output = module.forward([sample])
    pred = output[0].argmax(dim=1).item()
    if pred == y_encoded[i]:
        correct += 1

print(f"Accuracy : {correct/total*100:.2f}% ({correct}/{total})")

# ── Inférence sur un sample ───────────────────────────────────────
sample = torch.tensor(x_scaled[0:1], dtype=torch.float32)
output = module.forward([sample])
probs = output[0][0]

labels = le.classes_  # ['B', 'M']
print(f"\nSample 0 → vrai label : {y.iloc[0]}")
for i, label in enumerate(labels):
    print(f"  {label} : {probs[i]*100:.1f}%")
print(f"→ Prédit : {labels[probs.argmax().item()]}")