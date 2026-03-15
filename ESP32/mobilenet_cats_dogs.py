import emlearn_cnn_int8 as emlearn_cnn
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
            if line: vals.append(int(line))
    return vals

def predict(raw):
    out = array.array('f', (-1 for _ in range(out_length)))
    x = array.array('B', raw)
    print("Predict")
    model.run(x, out)
    print(out)
    return argmax(out)

def load_img(path):
    with open(path, "rb") as f:
        data = f.read()
        img = array.array("B", data)
    return img

print("Loading model")
with open("mobilenet_int8.tmdl", 'rb') as f:
    model_data = array.array('B', f.read())
model = emlearn_cnn.new(model_data)
out_length = model.output_dimensions()[0]

raw = load_img("cat-96.bin")  
predicted = predict(raw)
print(predicted)


