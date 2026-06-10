import emlearn_cnn_fp32 as emlearn_cnn
import array

def argmax(arr):
    idx_max = 0
    value_max = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > value_max:
            value_max = arr[i]
            idx_max = i
    return idx_max

def load_csv_1d(f):
    vals = []
    with open(f) as fp:
        for line in fp:
            line = line.strip()
            if line: vals.append(float(line))
    return vals

def scale(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)
      
def normalize(row):
    print("Normalize")
    l = [int(scale(row[i], scaler_min[i], scaler_max[i]) * 256) for i in range(len(row))]
    return array.array('B', l)    
        
def predict(raw):
    out = array.array('f', (-1 for _ in range(out_length)))
    x = normalize(raw)
    print("Predict")
    model.run(x, out)
    print(out)
    return argmax(out)

print("Loading scaler")
scaler_min  = load_csv_1d('scaler_cancer_min.csv')
scaler_max = load_csv_1d('scaler_cancer_max.csv')

print("Loading model")
with open("cancer_mlp.tmdl", 'rb') as f:
    model_data = array.array('B', f.read())
model = emlearn_cnn.new(model_data)
out_length = model.output_dimensions()[0]

data = [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]
pred = predict(data)
print(pred)
data = [13,21.82,87.5,519.8,0.1273,0.1932,0.1859,0.09353,0.235,0.07389,0.3063,1.002,2.406,24.32,0.005731,0.03502,0.03553,0.01226,0.02143,0.003749,15.49,30.73,106.2,739.3,0.1703,0.5401,0.539,0.206,0.4378,0.1072]
pred = predict(data)
print(pred)
data = [11.51,23.93,74.52,403.5,0.09261,0.1021,0.1112,0.04105,0.1388,0.0657,0.2388,2.904,1.936,16.97,0.0082,0.02982,0.05738,0.01267,0.01488,0.004738,12.48,37.16,82.28,474.2,0.1298,0.2517,0.363,0.09653,0.2112,0.08732]
pred = predict(data)
print(pred)

