import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=FutureWarning)

import serial
import numpy as np
import time

PORT = "COM3"      # Pico via USB CDC (COM virtual)
BAUD = 115200
DIGITO = 7         # troque aqui (0..9)

def load_mnist_samples():
    try:
        from tensorflow.keras.datasets import mnist # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow/Keras nao encontrado. Instale para usar MNIST: pip install tensorflow"
        ) from exc

    (x_train, y_train), _ = mnist.load_data()
    imgs = {}
    for digit in range(10):
        idx = int(np.where(y_train == digit)[0][0])
        imgs[digit] = x_train[idx].astype(np.uint8).flatten().tolist()
    return imgs


imgs = load_mnist_samples()

img = np.array(imgs[DIGITO], dtype=np.uint8)
assert img.size == 784

with serial.Serial(PORT, BAUD, timeout=5) as ser:
    time.sleep(2)
    ser.reset_input_buffer()
    line = ser.readline().decode(errors="ignore").strip()
    if line:
        print(line)
    ser.write(img.tobytes() + bytes([DIGITO]))  # 784 bytes + label
    ser.flush()
    for _ in range(10):
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print(line)
        if line.startswith("Prediction:"):
            break
