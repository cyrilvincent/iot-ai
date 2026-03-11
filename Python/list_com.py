import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
print("Ports lists")
for port in ports:
    print(port)
