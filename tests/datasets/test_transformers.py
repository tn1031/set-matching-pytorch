import numpy as np

from set_matching.datasets.transforms import ImageListTransform


def test_image_list_transform():
    channel, insize = 3, 299
    transform = ImageListTransform(
        "inception_v3", max_set_size=7, apply_flip=True, apply_shuffle=True, apply_padding=True
    )
    x = [np.random.rand(channel * insize * insize).reshape(channel, insize, insize) for _ in range(4)]

    y, mask = transform(x)
    assert y.shape == (7, channel, insize, insize)
    assert np.all(mask == np.array([True, True, True, True, False, False, False]))
