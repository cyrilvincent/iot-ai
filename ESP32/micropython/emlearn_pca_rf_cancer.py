import emlearn_trees
import array
from pca_lite import PCALite

def load_csv_1d(f):
    vals = []
    with open(f) as fp:
        for line in fp:
            line = line.strip()
            if line: vals.append(float(line))
    return vals

print("Loading scaler")
scaler_mean  = load_csv_1d('scaler_pca_cancer_mean.csv')
scaler_scale = load_csv_1d('scaler_pca_cancer_scale.csv')

nb_tree = 10
max_nodes = 400
features = 2
model = emlearn_trees.new(nb_tree, max_nodes, features)

print("Loading model")
with open('cancer_pca_rf_model.csv', 'r') as f:
    emlearn_trees.load_model(model, f)
print("Loaded")   
        
def normalize(row):
    print("Normalize")
    return array.array('h', [
        int((row[i] - scaler_mean[i]) / scaler_scale[i] * 32767)
        for i in range(len(row))
    ])    
        
def predict(raw):
    out = array.array('f', range(model.outputs()))
    x = normalize(raw)
    print("Predict")
    model.predict(x, out)
    print(out)
    return 0 if out[0] > out[1] else 1

pca = PCALite(10, 5)
data = [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]
data = pca.reduce1(data)
pred = predict(data)
print(pred)
data = [7.76,24.54,47.92,181,0.05263,0.04362,0,0,0.1587,0.05884,0.3857,1.428,2.548,19.15,0.007189,0.00466,0,0,0.02676,0.002783,9.456,30.37,59.16,268.6,0.08996,0.06444,0,0,0.2871,0.07039]
data = pca.reduce1(data)
pred = predict(data)
print(pred)

