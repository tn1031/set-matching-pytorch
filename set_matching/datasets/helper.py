import random

import numpy as np
from PIL import Image


def read_image(path, root, insize):
    f = Image.open(root / path)
    try:
        img = f.convert("RGB")
        img = img.resize((insize, insize), Image.ANTIALIAS)
        img = np.asarray(img, dtype=np.float32)
    finally:
        if hasattr(f, "close"):
            f.close()
    return img.transpose(2, 0, 1)


def random_flip(img, y_random=False, x_random=False, return_param=False, copy=False):
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {"y_flip": y_flip, "x_flip": x_flip}
    else:
        return img


def rescale_inception(image):
    img = image / 255.0
    img[0, :] = (img[0, :] - 0.5) / 0.5
    img[1, :] = (img[1, :] - 0.5) / 0.5
    img[2, :] = (img[2, :] - 0.5) / 0.5
    return img


def rescale_resnet(image):
    img = image / 255.0
    img[0, :] = (img[0, :] - 0.485) / 0.229
    img[1, :] = (img[1, :] - 0.456) / 0.224
    img[2, :] = (img[2, :] - 0.406) / 0.225
    return img
