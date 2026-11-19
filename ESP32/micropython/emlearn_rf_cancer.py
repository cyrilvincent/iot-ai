import emlearn_trees
import array

def load_csv_1d(f):
    vals = []
    with open(f) as fp:
        for line in fp:
            line = line.strip()
            if line: vals.append(float(line))
    return vals

print("Loading scaler")
scaler_mean  = load_csv_1d('scaler_cancer_mean.csv')
scaler_scale = load_csv_1d('scaler_cancer_scale.csv')

nb_tree = 20
max_nodes = 1000
features = 2
model = emlearn_trees.new(nb_tree, max_nodes, features)

print("Loading model")
with open('cancer_rf_model.csv', 'r') as f:
    emlearn_trees.load_model(model, f)
print("Loaded")   
        
def normalize(row):
    print("Normalize")
    return array.array('h', [
        int((row[i] - scaler_mean[i]) / scaler_scale[i] * 32767 / 3)
        for i in range(len(row))
    ])    
        
def predict(raw):
    out = array.array('f', range(model.outputs()))
    x = normalize(raw)
    print("Predict")
    model.predict(x, out)
    print(out)
    return 0 if out[0] > out[1] else 1

data = [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]
pred = predict(data)
print(pred)
data = [13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]
pred = predict(data)
print(pred)

