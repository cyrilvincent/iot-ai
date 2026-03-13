from machine import Pin
import time

p2 = Pin(2, Pin.OUT)
boot = Pin(0, Pin.IN, Pin.PULL_UP)
on = False

print("Started")
p2.off()
while True:
    if boot.value() == 0:
        on = not(on)
        print(on)
        if on:
            p2.on()
        else:
            p2.off()
        time.sleep_ms(500)
        
        
