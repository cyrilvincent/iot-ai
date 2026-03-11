import http.client
import json

conn = http.client.HTTPConnection("192.168.1.210:5000")
conn.request("GET", "/")
r1 = conn.getresponse()
print(r1.status, r1.reason)
data = r1.read().decode()
print(data)
conn.request("GET", "/all")
r1 = conn.getresponse()
print(r1.status, r1.reason)
data = r1.read().decode()
print(data)
data = json.loads(data)
print(data)
print(data["temp"])
