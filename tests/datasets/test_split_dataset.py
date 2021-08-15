import json
import pathlib

import numpy as np
import pytest

from set_matching.datasets.split_dataset import SplitDataset, FIMBsDataset


def test_split_dataset(data_dir):
    if data_dir is None:
        pytest.skip("there is no testdata.")

    root = pathlib.Path(".")
    data = json.load(open(pathlib.Path(data_dir) / "iqon_valid.json"))
    max_set_size = 6
    dataset = SplitDataset(data, root, 1, "resnet18", max_set_size, is_train=True)

    for i in range(min(len(data), 5)):
        y_size = min(len(data[i]["items"]) // 2, max_set_size)
        x_size = min(len(data[i]["items"]) - len(data[i]["items"]) // 2, max_set_size)
        x_images, x_mask, y_images, y_mask = dataset[i]

        assert x_images.shape == (max_set_size, 3, 224, 224)
        assert np.all(x_mask == np.array([True] * x_size + [False] * (max_set_size - x_size)))
        assert y_images.shape == (max_set_size, 3, 224, 224)
        assert np.all(y_mask == np.array([True] * y_size + [False] * (max_set_size - y_size)))

    max_set_size = 8  # if min_set_size=4, superset size = n_mix * (min_set_size // 2) >= max_set_size(=8)
    dataset = SplitDataset(data, root, 4, "resnet18", max_set_size, is_train=True)

    for i in range(min(len(data), 5)):
        y_size = min(len(data[i]["items"]) // 2, max_set_size)
        x_size = min(len(data[i]["items"]) - len(data[i]["items"]) // 2, max_set_size)
        x_images, x_mask, y_images, y_mask = dataset[i]

        assert x_images.shape == (max_set_size, 3, 224, 224)
        assert np.all(x_mask == np.array([True] * max_set_size))
        assert y_images.shape == (max_set_size, 3, 224, 224)
        assert np.all(y_mask == np.array([True] * max_set_size))


def test_fimbs_dataset(data_dir):
    if data_dir is None:
        pytest.skip("there is no testdata.")

    root = pathlib.Path(".")
    data = json.load(open(pathlib.Path(data_dir) / "iqon_test_finbs.json"))
    max_set_size_query, max_set_size_answer = 8, 2

    dataset = FIMBsDataset(
        data, root, cnn_arch="resnet18", max_set_size_query=max_set_size_query, max_set_size_answer=max_set_size_answer
    )

    for i in range(min(len(data), 5)):
        n_query_items = min(len(data[i]["query"]), max_set_size_query)
        query, q_mask, answers, a_mask = dataset[i]

        assert query.shape == (max_set_size_query, 3, 224, 224)
        assert np.all(q_mask == np.array([True] * n_query_items + [False] * (max_set_size_query - n_query_items)))
        assert answers.shape[1:] == (max_set_size_answer, 3, 224, 224)
        assert a_mask.shape[1] == max_set_size_answer
