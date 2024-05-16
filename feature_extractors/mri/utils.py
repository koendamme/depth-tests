import numpy as np


def preprocess_image(img):
    img = np.array(img)
    return np.uint8(img)