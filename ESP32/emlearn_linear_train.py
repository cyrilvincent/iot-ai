# List of build mpy
# https://github.com/emlearn/emlearn-micropython/tree/gh-pages/builds

# main.py — sur l'ESP32 en MicroPython
import emlearn_linreg
import array

# --- Lecture du CSV ---
def read_csv(filename):
    surfaces = array.array('f')
    loyers = array.array('f')
    with open(filename, 'r') as f:
        f.readline()  # skip header
        for row in f:
            row = row.strip()
            cols = row.split(',')
            surfaces.append(float(cols[0]))
            loyers.append(float(cols[1]))
    return surfaces, loyers

surfaces, loyers = read_csv('house.csv')
n = len(surfaces)
print(f"Données chargées : {n} lignes")

s_min = min(surfaces)
s_max = max(surfaces)
l_min = min(loyers)
l_max = max(loyers)

print(s_min, s_max, l_min, l_max)

def norm_s(v): return (v - s_min) / (s_max - s_min)
def norm_l(v): return (v - l_min) / (l_max - l_min)
def denorm_l(v): return v * (l_max - l_min) + l_min

# n_features=1, alpha=0.0001, l1_ratio=0.15, learning_rate=0.0001
model = emlearn_linreg.new(1, 0.1, 0.15, 0.0001)

x = array.array('f', [norm_s(s) for s in surfaces])
y = array.array('f', [norm_l(l) for l in loyers])
emlearn_linreg.train(model, x, y, verbose=1, max_iterations=1000)

print("Entraînement terminé")

# --- Prédictions ---
tests = [10.0, 50.0, 75.0, 100.0]
for surface in tests:
    x = array.array('f', [norm_s(surface)])
    out = model.predict(x)
    print(out)
    print(denorm_l(out))