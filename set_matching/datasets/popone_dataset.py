import json
import os
import pathlib

import numpy as np
import torch

from set_matching.datasets.helper import read_image
from set_matching.datasets.transforms import ImageListTransform, SingleImageTransform


def get_loader(fname, data_dir, batch_size, cnn_arch, max_set_size=8, is_train=True, num_workers=None):
    root = pathlib.Path(data_dir)
    data = json.load(open(root / fname))

    dataset = PopOneDataset(data, root, cnn_arch, max_set_size=max_set_size - 1, is_train=is_train)
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers if num_workers else os.cpu_count(),
        drop_last=True,
    )
    return loader


def get_fitb_loader(fname, data_dir, batch_size, cnn_arch, max_set_size=8, num_workers=None):
    root = pathlib.Path(data_dir)
    test = json.load(open(root / fname))
    test = [s for s in test if len(s["question"]) <= max_set_size]

    test_dataset = FITBDataset(test, root, cnn_arch=cnn_arch, max_set_size=max_set_size - 1)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers if num_workers else os.cpu_count(),
        drop_last=False,
    )
    return test_loader


class PopOneDataset(torch.utils.data.Dataset):
    def __init__(self, sets, root, cnn_arch, max_set_size, is_train):
        self.sets = sets
        self.root = root
        self.insize = 299 if cnn_arch == "inception_v3" else 224
        self.query_transform = ImageListTransform(
            cnn_arch, max_set_size=max_set_size, apply_flip=is_train, apply_shuffle=True, apply_padding=True
        )
        self.target_transform = SingleImageTransform(cnn_arch, apply_flip=is_train)

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        _set = self.sets[idx]
        items = _set["items"]
        images = [read_image(item, self.root, self.insize) for item in items]

        pop_idx = np.random.choice(len(images))
        target = images.pop(pop_idx)

        images, mask = self.query_transform(images)
        target = self.target_transform(target)
        return images, mask, target


class FITBDataset(torch.utils.data.Dataset):
    def __init__(self, sets, root, cnn_arch, max_set_size):
        self.sets = sets
        self.root = root
        self.insize = 299 if cnn_arch == "inception_v3" else 224
        self.query_transform = ImageListTransform(
            cnn_arch, max_set_size=max_set_size, apply_flip=False, apply_shuffle=False, apply_padding=True
        )
        self.target_transform = ImageListTransform(
            cnn_arch, max_set_size=4, apply_flip=False, apply_shuffle=False, apply_padding=False
        )

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        _set = self.sets[idx]
        question = [read_image(item, self.root, self.insize) for item in _set["question"]]
        _ = question.pop(_set["blank_position"])
        answer = [read_image(item, self.root, self.insize) for item in _set["answer"]]

        question, mask = self.query_transform(question)
        answer, _ = self.target_transform(answer)

        return question, mask, answer
