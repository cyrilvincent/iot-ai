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
scaler_mean  = load_csv_1d('scaler_mean.csv')
scaler_scale = load_csv_1d('scaler_scale.csv')

nb_tree = 3
max_nodes = 50
features = 2
model = emlearn_trees.new(nb_tree, max_nodes, features)

print("Loading model")
with open('heart_rf_model.csv', 'r') as f:
    emlearn_trees.load_model(model, f)
print("Loaded")   
        
def normalize(row):
    print("Normalize")
    return array.array('h', [
        int(((row[i] - scaler_mean[i]) / scaler_scale[i]) * 32767)
        for i in range(len(row))
    ])    
        
def predict(raw):
    out = array.array('f', range(model.outputs()))
    x = normalize(raw)
    print("Predict")
    model.predict(x, out)
    print(out)
    return 0 if out[0] > out[1] else 1

data = [28,1,2,130,132,0,2,185,0,0]
pred = predict(data)
print(pred)
data = [65,1,4,130,275,0,1,115,1,1]
pred = predict(data)
print(pred)

