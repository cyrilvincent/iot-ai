import machine
import i2c_lcd1602
import time

print("LCD I2C")
i2c = machine.I2C(sda=machine.Pin(21), scl=machine.Pin(22))

lcd = i2c_lcd1602.I2C_LCD1602(i2c)
lcd.puts("I2C LCD1602")

n = 0
while True:
    print(n)
    lcd.puts(n, 0, 1)
    n += 1
    time.sleep_ms(1000)

