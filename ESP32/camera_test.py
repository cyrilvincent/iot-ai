import camera

cam = camera.Camera(
    data_pins=[4, 5, 18, 19, 36, 39, 34, 35],
    vsync_pin=25,
    href_pin=23,
    sda_pin=26,
    scl_pin=27,
    xclk_pin=21,   # ← était 0
    pclk_pin=22,   # ← était 21
)

print("Caméra initialisée ✓")