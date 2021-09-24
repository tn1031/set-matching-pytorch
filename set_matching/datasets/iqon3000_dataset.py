import json
import os
import pathlib

import numpy as np
import torch
from set_matching.datasets.helper import read_image
from set_matching.datasets.transforms import ImageListTransform, SingleImageTransform


def get_loader(
    task_name, fname, data_dir, batch_size, embedder_arch, max_set_size=8, is_train=True, num_workers=None, **kwargs
):
    root = pathlib.Path(data_dir)
    data = json.load(open(root / fname))

    dataset_class = {"set_transformer": PopOneDataset, "set_matching": SplitDataset}[task_name]
    extra_config = {}
    if task_name == "set_matching":
        extra_config["n_mix"] = kwargs["n_mix"]
    dataset = dataset_class(
        data, root, cnn_arch=embedder_arch, max_set_size=max_set_size - 1, is_train=is_train, **extra_config
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers if num_workers else os.cpu_count(),
        drop_last=True,
    )
    return loader


def get_fitb_loader(fname, data_dir, batch_size, embedder_arch, max_set_size=8, num_workers=None):
    root = pathlib.Path(data_dir)
    test = json.load(open(root / fname))
    test = [s for s in test if len(s["question"]) <= max_set_size]

    test_dataset = FITBDataset(test, root, cnn_arch=embedder_arch, max_set_size=max_set_size - 1)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers if num_workers else os.cpu_count(),
        drop_last=False,
    )
    return test_loader


def get_finbs_loader(
    fname, data_dir, batch_size, embedder_arch, max_set_size_query=8, max_set_size_answer=2, num_workers=None
):
    root = pathlib.Path(data_dir)
    test = json.load(open(root / fname))

    test_dataset = FINBsDataset(
        test,
        root,
        cnn_arch=embedder_arch,
        max_set_size_query=max_set_size_query,
        max_set_size_answer=max_set_size_answer,
    )
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
        images = [read_image(item["path"], self.root, self.insize) for item in items]

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
        question = [read_image(item["path"], self.root, self.insize) for item in _set["question"]]
        _ = question.pop(_set["blank_position"])
        answer = [read_image(item["path"], self.root, self.insize) for item in _set["answer"]]

        question, mask = self.query_transform(question)
        answer, _ = self.target_transform(answer)

        return question, mask, answer


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, sets, root, *, n_mix, cnn_arch, max_set_size, is_train) -> None:
        self.sets = sets
        self.root = root
        self.n_mix = n_mix
        self.insize = 299 if cnn_arch == "inception_v3" else 224
        self.query_transform = ImageListTransform(
            cnn_arch, max_set_size=max_set_size, apply_flip=is_train, apply_shuffle=True, apply_padding=True
        )

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        if self.n_mix > 1:
            indices = np.delete(np.arange(len(self.sets)), idx)
            indices = np.random.choice(indices, self.n_mix - 1, replace=False)
            indices = [idx] + list(indices)
        else:
            indices = [idx]

        x_images, y_images = [], []
        for i in indices:
            _set = self.sets[i]
            items = _set["items"]
            images = [read_image(item["path"], self.root, self.insize) for item in items]
            images = np.array(images)

            y_size = len(images) // 2

            xy_mask = [True] * (len(images) - y_size) + [False] * y_size
            xy_mask = np.random.permutation(xy_mask)
            x_images.extend(list(images[xy_mask, :]))
            y_images.extend(list(images[~xy_mask, :]))

        x_images, x_mask = self.query_transform(x_images)
        y_images, y_mask = self.query_transform(y_images)
        return x_images, x_mask, y_images, y_mask


class FINBsDataset(torch.utils.data.Dataset):
    def __init__(self, sets, root, cnn_arch, max_set_size_query, max_set_size_answer):
        self.sets = sets
        self.root = root
        self.insize = 299 if cnn_arch == "inception_v3" else 224
        self.transform_q = ImageListTransform(
            cnn_arch, max_set_size=max_set_size_query, apply_flip=False, apply_shuffle=False, apply_padding=True
        )
        self.transform_a = ImageListTransform(
            cnn_arch, max_set_size=max_set_size_answer, apply_flip=False, apply_shuffle=False, apply_padding=True
        )

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        _set = self.sets[idx]
        question = [read_image(item["path"], self.root, self.insize) for item in _set["query"]]
        _answers = []
        for cand in _set["answers"]:
            cand_images = [read_image(item["path"], self.root, self.insize) for item in cand]
            _answers.append(cand_images)

        question, q_mask = self.transform_q(question)
        answers, a_masks = [], []
        for answer in _answers:
            ans, a_mask = self.transform_a(answer)
            answers.append(ans)
            a_masks.append(a_mask)

        return question, q_mask, np.array(answers), np.array(a_masks)
