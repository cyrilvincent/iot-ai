import time
import machine
import network

ssid = "StarlinkVincent"
key = "famillevincent"

def scan():
    print("Scanning Wifi")
    wlan = network.WLAN()
    wlan.active(True)
    print("Scanned Wifi:", wlan.scan())

def wifi_connect(ssid, key):
    wlan = network.WLAN()
    print("Activate Wifi")
    wlan.active(True)
    if not wlan.isconnected():
        print(f'connecting to Wifi {ssid}')
        wlan.connect(ssid, key)
        while not wlan.isconnected():
            machine.idle()
    print(f'Connected to wifi {ssid}')    
    print('Wifi config:', wlan.ipconfig('addr4'))

if __name__ == "__main__":
    p2 = machine.Pin(2, machine.Pin.OUT)
    p2.off()
    scan() 
    print(f"Connecting to Wifi")
    wifi_connect(ssid, key)
    p2.on()

    


