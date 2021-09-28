import numpy as np
from set_matching.datasets.transforms import FeatureListTransform, ImageListTransform


def test_image_list_transform():
    channel, insize = 3, 299
    transform = ImageListTransform(
        "inception_v3", max_set_size=7, apply_flip=True, apply_shuffle=True, apply_padding=True
    )
    x = [np.random.rand(channel * insize * insize).reshape(channel, insize, insize) for _ in range(4)]

    y, mask = transform(x)
    assert y.shape == (7, channel, insize, insize)
    assert np.all(mask == np.array([True, True, True, True, False, False, False]))


def test_feature_list_transform():
    transform = FeatureListTransform(max_set_size=7, apply_shuffle=True, apply_padding=True)
    x = [np.random.rand(128) for _ in range(4)]
    c = [1, 2, 3, 4]

    y, y_category, y_mask = transform(x, c)
    assert y.shape == (7, 128)
    assert y_category.shape == (7,)
    assert np.all(y_category[-3:] == np.array([0, 0, 0]))
    assert np.all(y_mask == np.array([True, True, True, True, False, False, False]))

    x = [np.random.rand(128) for _ in range(8)]
    c = [1, 2, 3, 4, 1, 2, 3, 4]

    y, y_category, y_mask = transform(x, c)
    assert y.shape == (7, 128)
    assert y_category.shape == (7,)
    assert np.all(y_category != np.zeros(7))
    assert np.all(y_mask == np.array([True, True, True, True, True, True, True]))
