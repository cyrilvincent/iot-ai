import time
import machine
import esp32
import sys

def convert_fahrenheit_to_celsius(fahrenheit):
    celsius = float(fahrenheit - 32) * 5 / 9
    return celsius

# machine.freq(160000000) # for wroom
machine.freq(240000000) # for vrower
print(f"Python version: {sys.version}")
rtc = machine.RTC()
print(f"DateTime: {rtc.datetime()}")
print(f"CPU Frequency: {machine.freq() // 1000000}MHz")
while True:
    print(f"CPU Temperature: {int(convert_fahrenheit_to_celsius(esp32.raw_temperature()))}°C")
    time.sleep(10)
    


