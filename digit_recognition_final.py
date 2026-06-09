import cv2
import numpy as np
import tensorflow as tf
from sense_hat import SenseHat
import time

# ── Setup ──────────────────────────────────────────
sense = SenseHat()
model = tf.keras.models.load_model('model_mnist_cnn.h5', compile=False)
cam = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cam.isOpened():
    print("No se detectó cámara.")
    exit()

print("Listo. Muestra un dígito a la cámara. Presiona ESC para salir.")

# ── Colores para el Sense HAT ───────────────────────
GREEN  = [0, 255, 0]
RED    = [255, 0, 0]
BLUE   = [0, 0, 255]
YELLOW = [255, 255, 0]
OFF    = [0, 0, 0]

# Un color diferente por dígito
DIGIT_COLORS = {
    0: [255, 255, 255],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
    4: [255, 255, 0],
    5: [255, 0, 255],
    6: [0, 255, 255],
    7: [255, 128, 0],
    8: [128, 0, 255],
    9: [0, 255, 128]}
