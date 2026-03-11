import time
import machine

print("Blinking")
p2 = machine.Pin(25, machine.Pin.OUT)

while True:
    p2.on()
    time.sleep_ms(750)
    p2.off()
    time.sleep_ms(250)