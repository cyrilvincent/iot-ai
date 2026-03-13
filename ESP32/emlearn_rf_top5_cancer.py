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
with open('cancer_rf_top5_model.csv', 'r') as f:
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

data = [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]
data = [data[7], data[2], data[23], data[20], data[22]]
pred = predict(data)
print(pred)
data = [13,21.82,87.5,519.8,0.1273,0.1932,0.1859,0.09353,0.235,0.07389,0.3063,1.002,2.406,24.32,0.005731,0.03502,0.03553,0.01226,0.02143,0.003749,15.49,30.73,106.2,739.3,0.1703,0.5401,0.539,0.206,0.4378,0.1072]
data = [data[7], data[2], data[23], data[20], data[22]]
pred = predict(data)
print(pred)

