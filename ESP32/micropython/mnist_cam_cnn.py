import emlearn_cnn_int8 as emlearn_cnn
import array
from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling
from machine import Pin
import time

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
            
def reduce92_28(raw):
    mnist=[]
    for x in range(0,28):
        for y in range(0,28):
            value = raw[int(96/28*x)*96+int(96/28*y)] + 128
            mnist.append(value)
    return mnist
            
def capture():
    print("Camera capture")
    cam = Camera(pixel_format=PixelFormat.GRAYSCALE, frame_size=FrameSize.R96X96)
    raw = list(cam.capture())
    mnist = reduce92_28(raw)
    mnist = [255 if x > 128 else 0 for x in mnist]
    with open("cam.csv","w") as f:
        for i in mnist:
            f.write(f"{i}\n")

print("Loading model")
with open("mnist_cnn_int8.tmdl", 'rb') as f:
    model_data = array.array('B', f.read())
model = emlearn_cnn.new(model_data)
out_length = model.output_dimensions()[0]

path="cam.csv"
while True:
    capture()
    display_csv(path)
    raw = load_csv_1d(path)
    predicted = predict(raw)
    print(predicted)
    time.sleep(1)
    


