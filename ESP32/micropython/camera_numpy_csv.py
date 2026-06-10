from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling
import time

# https://github.com/cnadler86/micropython-camera-API

cam = Camera(pixel_format=PixelFormat.GRAYSCALE, frame_size=FrameSize.R96X96)
print("Caméra initialisée ✓")

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

def crop28(raw):
    mnist=[]
    for x in range(48-14,48+14):
        for y in range(48-14,48+14):
            value = raw[x*96+y] + 128
            mnist.append(value)
    return mnist        
            
def capture():
    cam = Camera(pixel_format=PixelFormat.GRAYSCALE, frame_size=FrameSize.R96X96)
    raw = list(cam.capture())
    mnist = reduce92_28(raw)
    # mnist = crop28(raw)
    print(len(mnist))
    mnist = [255 if x > 128 else 0 for x in mnist]
    with open("cam.csv","w") as f:
        for i in mnist:
            f.write(f"{i}\n")
           
while True:
    capture()
    display_csv("cam.csv")
    time.sleep(1)
        



