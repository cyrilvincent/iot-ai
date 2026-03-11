from microdot import Microdot
import bme280_float as bme280
import machine
import json
import i2c_lcd

# Wifi must be activated

app = Microdot()
i2c = machine.SoftI2C(sda=machine.Pin(21), scl=machine.Pin(22))
bme = None # bme280.BME280(i2c=i2c)
lcd = None # i2c_lcd.Display(i2c)

@app.route("/")
async def index(request):
    return "Hello from ESP32"

@app.route("/temp")
async def temp(request):
    temp, pre, hum = bme.read_compensated_data(result = None)
    result = {"temp":temp}
    print(result)
    return result

@app.route("/temp/<param>")
async def temp_param(request, param):
    temp, pre, hum = bme.read_compensated_data(result = None)
    if param == "c":
        result = {"temp":temp}
    elif param == "f":
        result = {"temp":temp * 9/5 + 32}
    else:
        result = {"temp": f"error {param}"}
    print(result)
    return result

@app.route("/hum")
async def hum(request):
    temp, pre, hum = bme.read_compensated_data(result = None)
    result = {"hum":hum}
    print(result)
    return result

@app.route("/all")
async def hum(request):
    temp, pre, hum = bme.read_compensated_data(result = None)
    result = {"temp": temp, "hum":hum, "pre":pre}
    print(result)
    return result


@app.route("/display/<param>")
async def display(request, param):
    print(param)
    lcd.move(0,0)
    lcd.write(param + "            ")
    return param

app.run(debug=True)
# 192.168.1.210