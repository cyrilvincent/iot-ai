from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling

# https://github.com/cnadler86/micropython-camera-API

cam = Camera(pixel_format=PixelFormat.JPEG)
print("Caméra initialisée ✓")
cam.init()
img = cam.capture()
with open("cam.jpg", "wb") as f:
   f.write(img)