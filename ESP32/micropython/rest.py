from microdot import Microdot

app = Microdot()


@app.route('/')
async def index(request):
    return 'Hello, world!'

@app.route("/toto")
async def toto(request):
    return f"return {request.query_string['key']}"

app.run(debug=True)
# 192.168.1.210
# Wifi must be activated