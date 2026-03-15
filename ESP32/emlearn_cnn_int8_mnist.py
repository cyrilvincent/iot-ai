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

def display_csv(path):
    with open(path) as f:
        rows = list(f)
        for y in range(28):
            for x in range(28):
                v = int(rows[x + y*28].strip())
                print(f"{' ' if v < 128 else 'O'}", end="")
            print()

print("Loading model")
with open("mnist_cnn_int8.tmdl", 'rb') as f:
    model_data = array.array('B', f.read())
model = emlearn_cnn.new(model_data)
out_length = model.output_dimensions()[0]

for nb in range(10):
    display_csv(f"{nb}.csv")
    raw = load_csv_1d(f"{nb}.csv")
    predicted = predict(raw)
    print(predicted)


