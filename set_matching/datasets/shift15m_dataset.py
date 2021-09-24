import gzip
import json
import os
import pathlib

import numpy as np
import torch
from set_matching.datasets.transforms import FeatureListTransform

CATEGORIES = {c: i + 1 for i, c in enumerate("10,11,12,13,14,15,16".split(","))}  # 0 is an ignore idx


def get_loader(task_name, fname, data_dir, batch_size, max_set_size=8, num_workers=None, **kwargs):
    root = pathlib.Path(data_dir)
    data = json.load(open(root / fname))

    dataset_class = {"set_transformer": PopOneDataset, "set_matching": SplitDataset, "set_prediction": SplitDataset}[
        task_name
    ]
    extra_config = {}
    if task_name == "set_matching":
        extra_config["n_mix"] = kwargs["n_mix"]
        extra_config["use_category"] = False
    elif task_name == "set_prediction":
        extra_config["n_mix"] = kwargs["n_mix"]
        extra_config["use_category"] = True
    dataset = dataset_class(data, root, max_set_size=max_set_size, **extra_config)
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers if num_workers else os.cpu_count(),
        drop_last=True,
    )
    return loader


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, sets, root, *, n_mix, max_set_size, use_category) -> None:
        self.sets = sets
        self.root = root
        self.n_mix = n_mix
        self.use_category = use_category
        self.query_transform = FeatureListTransform(max_set_size=max_set_size, apply_shuffle=True, apply_padding=True)

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        if self.n_mix > 1:
            indices = np.delete(np.arange(len(self.sets)), idx)
            indices = np.random.choice(indices, self.n_mix - 1, replace=False)
            indices = [idx] + list(indices)
        else:
            indices = [idx]

        x_features, y_features = [], []
        x_categories, y_categories = [], []
        for i in indices:
            _set = self.sets[i]
            items = _set["items"]
            features, categories = [], []
            for item in items:
                with gzip.open(self.root / item["path"], "r") as f:
                    feature = json.load(f)
                features.append(feature)
                categories.append(CATEGORIES[item["category"]])
            features = np.array(features, dtype=np.float32)
            categories = np.array(categories, dtype=np.int32)

            y_size = len(features) // 2

            xy_mask = [True] * (len(features) - y_size) + [False] * y_size
            xy_mask = np.random.permutation(xy_mask)
            x_features.extend(list(features[xy_mask, :]))
            y_features.extend(list(features[~xy_mask, :]))
            x_categories.extend(list(categories[xy_mask]))
            y_categories.extend(list(categories[~xy_mask]))

        x_features, x_categories, x_mask = self.query_transform(x_features, x_categories)
        y_features, y_categories, y_mask = self.query_transform(y_features, y_categories)

        if self.use_category:
            return x_features, x_mask, y_categories, y_features
        else:
            return x_features, x_mask, y_features, y_categories != 0


class PopOneDataset(torch.utils.data.Dataset):
    def __init__(self, sets, root, *, max_set_size):
        self.sets = sets
        self.root = root
        self.query_transform = FeatureListTransform(max_set_size=max_set_size, apply_shuffle=True, apply_padding=True)

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        _set = self.sets[idx]
        items = _set["items"]
        features, categories = [], []
        for item in items:
            with gzip.open(self.root / item["path"], "r") as f:
                feature = json.load(f)
            features.append(feature)
            categories.append(CATEGORIES[item["category"]])

        pop_idx = np.random.choice(len(features))
        target = features.pop(pop_idx)
        _ = categories.pop(pop_idx)

        features, _, mask = self.query_transform(features, categories)
        return features, mask, np.array(target, dtype=np.float32)
