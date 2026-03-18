from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling

# https://github.com/cnadler86/micropython-camera-API

cam = Camera(pixel_format=PixelFormat.GRAYSCALE, frame_size=FrameSize.R96X96)
print("Caméra initialisée ✓")
rgb = list(cam.capture())
print(min(rgb), max(rgb))
print(len(rgb), len(rgb) ** 0.5)
mnist=[]
for x in range(0,28):
    for y in range(0,28):
        value = rgb[int(96/28*x)*96+int(96/28*y)] + 128
        mnist.append(value)
        
with open("cam.csv","w") as f:
    for i in mnist:
        f.write(f"{i}\n")
    


