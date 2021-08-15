import random

import numpy as np

from set_matching.datasets.helper import random_flip, rescale_inception, rescale_resnet


class ImageListTransform:
    def __init__(self, cnn_arch, max_set_size=7, apply_flip=True, apply_shuffle=True, apply_padding=True):
        self.max_set_size = max_set_size
        self.apply_flip = apply_flip
        self.apply_shuffle = apply_shuffle
        self.apply_padding = apply_padding
        if cnn_arch == "inception_v3":
            self._rescale = rescale_inception
        else:
            self._rescale = rescale_resnet

    def __call__(self, in_data):
        images = []
        for image in in_data:
            if self.apply_flip:
                image = random_flip(image, x_random=True, return_param=False)
            image = self._rescale(image)
            images.append(image)

        if self.apply_shuffle:
            random.shuffle(images)

        if len(images) > self.max_set_size:
            images = images[: self.max_set_size]
        mask = [True] * len(images)

        if self.apply_padding:
            while len(images) < self.max_set_size:
                images.append(images[-1])
                mask.append(False)

        return np.array(images), np.array(mask)


class SingleImageTransform:
    def __init__(self, cnn_arch, apply_flip=True):
        self.apply_flip = apply_flip
        if cnn_arch == "inception_v3":
            self._rescale = rescale_inception
        else:
            self._rescale = rescale_resnet

    def __call__(self, in_data):
        image = in_data
        if self.apply_flip:
            image = random_flip(image, x_random=True, return_param=False)
        image = self._rescale(image)
        return image
