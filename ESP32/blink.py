import time
import machine

print("Blinking")
p2 = machine.Pin(2, machine.Pin.OUT)
p13 = machine.Pin(13, machine.Pin.OUT)

while True:
    p2.on()
    p13.off()
    time.sleep_ms(750)
    p2.off()
    p13.on()
    time.sleep_ms(250)