from machine import Pin
import time

boot = Pin(0, Pin.IN, Pin.PULL_UP)

while True:
    if boot.value() == 0:
        print("Bouton BOOT appuyé")
    time.sleep_ms(100)    
        
