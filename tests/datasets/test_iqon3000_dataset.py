import json
import pathlib

import numpy as np
import pytest
from set_matching.datasets.iqon3000_dataset import FINBsDataset, FITBDataset, PopOneDataset, SplitDataset


def test_popone_dataset(data_dir):
    if data_dir is None:
        pytest.skip("there is no testdata.")

    root = pathlib.Path(data_dir)
    data = json.load(open(pathlib.Path(data_dir) / "IQON3000_valid.json"))
    max_set_size = 7
    dataset = PopOneDataset(data, root, "resnet18", max_set_size=max_set_size, is_train=True)

    for i in range(min(len(data), 5)):
        n_query_items = min(len(data[i]["items"]) - 1, max_set_size)
        images, mask, target = dataset[i]

        assert images.shape == (max_set_size, 3, 224, 224)
        assert np.all(mask == np.array([True] * n_query_items + [False] * (max_set_size - n_query_items)))
        assert target.shape == (3, 224, 224)


def test_fitb_dataset(data_dir):
    if data_dir is None:
        pytest.skip("there is no testdata.")

    root = pathlib.Path(data_dir)
    data = json.load(open(pathlib.Path(data_dir) / "IQON3000_test_fitb.json"))
    max_set_size = 8
    data = [s for s in data if len(s["question"]) <= max_set_size]

    dataset = FITBDataset(data, root, cnn_arch="resnet18", max_set_size=max_set_size - 1)

    for i in range(min(len(data), 5)):
        n_query_items = min(len(data[i]["question"]) - 1, max_set_size)
        question, mask, answer = dataset[i]

        assert question.shape == (max_set_size - 1, 3, 224, 224)
        assert np.all(mask == np.array([True] * n_query_items + [False] * (max_set_size - 1 - n_query_items)))
        assert answer.shape == (4, 3, 224, 224)

        # input file validation
        _set = data[i]
        assert _set["question"][int(_set["blank_position"])] == _set["answer"][0]


def test_split_dataset(data_dir):
    if data_dir is None:
        pytest.skip("there is no testdata.")

    root = pathlib.Path(data_dir)
    data = json.load(open(pathlib.Path(data_dir) / "IQON3000_valid.json"))
    max_set_size = 6
    dataset = SplitDataset(data, root, n_mix=1, cnn_arch="resnet18", max_set_size=max_set_size, is_train=True)

    for i in range(min(len(data), 5)):
        y_size = min(len(data[i]["items"]) // 2, max_set_size)
        x_size = min(len(data[i]["items"]) - len(data[i]["items"]) // 2, max_set_size)
        x_images, x_mask, y_images, y_mask = dataset[i]

        assert x_images.shape == (max_set_size, 3, 224, 224)
        assert np.all(x_mask == np.array([True] * x_size + [False] * (max_set_size - x_size)))
        assert y_images.shape == (max_set_size, 3, 224, 224)
        assert np.all(y_mask == np.array([True] * y_size + [False] * (max_set_size - y_size)))

    max_set_size = 8  # if min_set_size=4, superset size = n_mix * (min_set_size // 2) >= max_set_size(=8)
    dataset = SplitDataset(data, root, n_mix=4, cnn_arch="resnet18", max_set_size=max_set_size, is_train=True)

    for i in range(min(len(data), 5)):
        y_size = min(len(data[i]["items"]) // 2, max_set_size)
        x_size = min(len(data[i]["items"]) - len(data[i]["items"]) // 2, max_set_size)
        x_images, x_mask, y_images, y_mask = dataset[i]

        assert x_images.shape == (max_set_size, 3, 224, 224)
        assert np.all(x_mask == np.array([True] * max_set_size))
        assert y_images.shape == (max_set_size, 3, 224, 224)
        assert np.all(y_mask == np.array([True] * max_set_size))


def test_finbs_dataset(data_dir):
    if data_dir is None:
        pytest.skip("there is no testdata.")

    root = pathlib.Path(data_dir)
    data = json.load(open(pathlib.Path(data_dir) / "IQON3000_test_finbs.json"))
    max_set_size_query, max_set_size_answer = 8, 2

    dataset = FINBsDataset(
        data, root, cnn_arch="resnet18", max_set_size_query=max_set_size_query, max_set_size_answer=max_set_size_answer
    )

    for i in range(min(len(data), 5)):
        n_query_items = min(len(data[i]["query"]), max_set_size_query)
        query, q_mask, answers, a_mask = dataset[i]

        assert query.shape == (max_set_size_query, 3, 224, 224)
        assert np.all(q_mask == np.array([True] * n_query_items + [False] * (max_set_size_query - n_query_items)))
        assert answers.shape[1:] == (max_set_size_answer, 3, 224, 224)
        assert a_mask.shape[1] == max_set_size_answer
