import serial
import numpy as np
import time

PORT = "COM3"      # Pico via USB CDC (COM virtual)
BAUD = 115200
DIGITO = 6         # troque aqui (0..9)

def _draw_seg(img, r0, c0, r1, c1, val=255):
    img[r0:r1, c0:c1] = val


def make_digit(d):
    img = np.zeros((28, 28), dtype=np.uint8)
    t = 3  # espessura
    m = 4  # margem
    top = (m, m, m + t, 28 - m)
    mid = (13 - t // 2, m, 13 - t // 2 + t, 28 - m)
    bot = (28 - m - t, m, 28 - m, 28 - m)
    ul = (m, m, 13, m + t)
    ur = (m, 28 - m - t, 13, 28 - m)
    ll = (13, m, 28 - m, m + t)
    lr = (13, 28 - m - t, 28 - m, 28 - m)

    segments = {
        0: [top, ul, ur, ll, lr, bot],
        1: [ur, lr],
        2: [top, ur, mid, ll, bot],
        3: [top, ur, mid, lr, bot],
        4: [ul, ur, mid, lr],
        5: [top, ul, mid, lr, bot],
        6: [top, ul, mid, ll, lr, bot],
        7: [top, ur, lr],
        8: [top, ul, ur, mid, ll, lr, bot],
        9: [top, ul, ur, mid, lr, bot],
    }

    for seg in segments[int(d)]:
        _draw_seg(img, *seg)

    return img.flatten().tolist()


img0 = make_digit(0)
img1 = make_digit(1)
img2 = make_digit(2)
img3 = make_digit(3)
img4 = make_digit(4)
img5 = make_digit(5)
img6 = make_digit(6)
img7 = make_digit(7)
img8 = make_digit(8)
img9 = make_digit(9)

imgs = {i: v for i, v in enumerate([img0, img1, img2, img3, img4, img5, img6, img7, img8, img9])}

img = np.array(imgs[DIGITO], dtype=np.uint8)
assert img.size == 784

with serial.Serial(PORT, BAUD, timeout=5) as ser:
    time.sleep(2)
    ser.reset_input_buffer()
    line = ser.readline().decode(errors="ignore").strip()
    if line:
        print(line)
    ser.write(img.tobytes())  # 784 bytes crus
    ser.flush()
    for _ in range(10):
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print(line)
        if line.startswith("Prediction:"):
            break
