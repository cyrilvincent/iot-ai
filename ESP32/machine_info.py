import time
import machine
import esp32
import sys
import gc
import uos

def convert_fahrenheit_to_celsius(fahrenheit):
    celsius = float(fahrenheit - 32) * 5 / 9
    return celsius

# machine.freq(160000000) # for wroom
# machine.freq(240000000)

print(f"Python version: {sys.version}")
rtc = machine.RTC()
print(f"DateTime: {rtc.datetime()}")
print(f"CPU Frequency: {machine.freq() // 1000000}MHz")
gc.collect()
free = gc.mem_free()
total = gc.mem_alloc() + gc.mem_free()
print(f"RAM totale : {total/1024:.1f} KB")
print(f"RAM libre  : {free/1024:.1f} KB")
stats = uos.statvfs('/')
block_size = stats[0]
free_blocks = stats[3]
total_blocks = stats[2]
free_kb = (free_blocks * block_size) / 1024
total_kb = (total_blocks * block_size) / 1024
print(f"Stockage total : {total_kb:.1f} KB")
print(f"Stockage libre : {free_kb:.1f} KB")
print(f"CPU Temperature: {int(convert_fahrenheit_to_celsius(esp32.raw_temperature()))}°C")

    


