import time
import machine
import select
import sys

print("UART USB")
poll_object = select.poll()
poll_object.register(sys.stdin,1)

print("Input> ", end="") # Type your order in the shell
while True:
    if poll_object.poll(0):
       s = sys.stdin.readline().strip()
       if len(s) > 0:
           print(f"hello {s}")
           print("Input> ", end="")
    time.sleep_ms(100)