name = "data/har/har_conv1d_int8.tflite"

with open(name, "rb") as f:
    data = f.read()

with open(name.replace(".tflite", ".h"), "w") as f:
    f.write("unsigned char model_tflite[] = {\n  ")
    hex_values = ", ".join(f"0x{b:02x}" for b in data)
    # Retour à la ligne tous les 12 octets pour la lisibilité
    bytes_list = [f"0x{b:02x}" for b in data]
    lines = [", ".join(bytes_list[i:i+12]) for i in range(0, len(bytes_list), 12)]
    f.write(",\n  ".join(lines))
    f.write(f"\n}};\nunsigned int model_tflite_len = {len(data)};\n")
