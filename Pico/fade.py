import time
import machine
import math

print("Led Fading")
pwm2 = machine.PWM(machine.Pin(25), freq=1000, duty_u16=65535)

while True:
    for i in range(0,65535,66):
        pwm2.duty_u16(i)
        time.sleep_ms(1)
    for i in range(65535,-1,-66):
        pwm2.duty_u16(i)
        time.sleep_ms(1)
    
    
    
