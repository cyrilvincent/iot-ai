import struct
from camera import Camera

def rgb565_to_bmp(data, width, height):
    row_size = width * 3
    padding = (4 - (row_size % 4)) % 4
    padded_row = row_size + padding
    pixel_data_size = padded_row * height
    file_size = 54 + pixel_data_size

    # File header
    header = struct.pack('<2sIHHI', b'BM', file_size, 0, 0, 54)
    # DIB header
    dib = struct.pack('<IiiHHIIiiII',
        40, width, -height, 1, 24,
        0, pixel_data_size, 2835, 2835, 0, 0
    )

    rows = bytearray()
    for y in range(height):
        row = bytearray()
        for x in range(width):
            i = (y * width + x) * 2
            pixel = (data[i] << 8) | data[i + 1]
            r = ((pixel >> 11) & 0x1F) << 3
            g = ((pixel >> 5)  & 0x3F) << 2
            b = ((pixel >> 0)  & 0x1F) << 3
            row += bytes([b, g, r])  # BMP = BGR
        rows += row + b'\x00' * padding

    return header + dib + rows

cam = Camera()
img_raw = cam.capture()
bmp = rgb565_to_bmp(bytes(img_raw), 160, 120)

with open('/cam.bmp', 'wb') as f:
    f.write(bmp)

print("Saved:", len(bmp), "bytes")