import json
import pathlib

import numpy as np
import pytest

from set_matching.datasets.popone_dataset import FITBDataset, PopOneDataset


def test_popone_dataset(data_dir):
    if data_dir is None:
        pytest.skip("there is no testdata.")

    root = pathlib.Path(".")
    data = json.load(open(pathlib.Path(data_dir) / "iqon_valid.json"))
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

    root = pathlib.Path(".")
    data = json.load(open(pathlib.Path(data_dir) / "iqon_test_fitb.json"))
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
