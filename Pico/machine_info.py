import time
import machine
import sys

def convert_fahrenheit_to_celsius(fahrenheit):
    celsius = float(fahrenheit - 32) * 5 / 9
    return celsius

print(f"Python version: {sys.version}")
rtc = machine.RTC()
print(f"DateTime: {rtc.datetime()}")
print(f"CPU Frequency: {machine.freq() // 1000000}MHz")
    


