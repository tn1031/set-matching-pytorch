import json
import pathlib
import random

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def validate_imagefile(path):
    try:
        _ = Image.open(path)
        return True
    except UnidentifiedImageError:
        return False


def make_fitb_problems(test_sets, n_candidates):
    all_items = set()
    for _set in test_sets:
        all_items = all_items | set(_set["items"])
    all_items = list(all_items)

    test = []
    for _set in tqdm(test_sets):
        items = _set["items"]
        pos = random.randrange(len(items))
        answer = [items[pos]]

        while len(answer) < n_candidates:
            cand = random.choice(all_items)
            if cand not in answer:
                answer.append(cand)

        test.append({"question": items, "blank_position": pos, "answer": answer})

    return test


def split_itemsets(sets, indices):
    x_items, y_items = [], []
    for i in indices:
        items = np.array(sets[i]["items"])

        y_size = len(items) // 2

        xy_mask = [True] * (len(items) - y_size) + [False] * y_size
        random.shuffle(xy_mask)
        xy_mask = np.array(xy_mask)
        x_items.extend(list(items[xy_mask]))
        y_items.extend(list(items[~xy_mask]))

    x_items, y_items = list(set(x_items)), list(set(y_items))
    random.shuffle(x_items)
    random.shuffle(y_items)
    return x_items, y_items


def make_fimbs_problems(test_sets, n_candidates, n_mix, max_set_size_x=8, max_set_size_y=2):
    assert n_candidates > 1

    test = []
    for i in tqdm(range(len(test_sets))):
        answers = []

        indices = [i]
        if n_mix > 1:
            indices = list(range(len(test_sets)))
            indices.remove(i)
            indices = random.sample(indices, n_mix - 1)
            indices = [i] + indices

        x_items, y_items = split_itemsets(test_sets, indices)
        if len(x_items) > max_set_size_x:
            x_items = x_items[:max_set_size_x]

        if len(y_items) > max_set_size_y:
            y_items = y_items[:max_set_size_y]
        answers.append(y_items)

        neg_candidates = [idx for idx in range(len(test_sets)) if idx not in indices]
        for _ in range(n_candidates - 1):
            neg_indices = random.sample(neg_candidates, n_mix)

            _, y_items = split_itemsets(test_sets, neg_indices)
            if len(y_items) > max_set_size_y:
                y_items = y_items[:max_set_size_y]
            answers.append(y_items)

        test.append({"query": x_items, "answers": answers})

    return test


def main(args):
    random.seed(1219)

    train_ratio, test_ratio = args.train_size, args.test_size
    valid_ratio = 1 - (train_ratio + test_ratio)
    assert train_ratio + valid_ratio + test_ratio == 1

    root = pathlib.Path(args.data_dir)

    iqon_all = []
    for user_dir in (root / "IQON3000").iterdir():
        for set_dir in user_dir.iterdir():
            set_id = set_dir.name
            items = [
                item for item in set_dir.glob("*.jpg") if item.stat().st_size > 0 and validate_imagefile(str(item))
            ]
            if len(items) < args.min_set_size:
                continue
            random.shuffle(items)
            iqon_all.append({"set_id": set_id, "items": list(map(str, items))})

    train, valid_test = train_test_split(iqon_all, train_size=train_ratio, random_state=12)
    valid, test = train_test_split(valid_test, train_size=valid_ratio / (valid_ratio + test_ratio), random_state=19)

    with open(root / "iqon_train.json", "w") as f:
        json.dump(train, f, indent=2)
    with open(root / "iqon_valid.json", "w") as f:
        json.dump(valid, f, indent=2)

    fitb = make_fitb_problems(test, args.n_candidates)

    with open(root / "iqon_test_fitb.json", "w") as f:
        json.dump(fitb, f, indent=2)

    fimbs = make_fimbs_problems(test, args.n_candidates, args.n_mix, args.max_set_size_x, args.max_set_size_y)
    with open(root / "iqon_test_finbs.json", "w") as f:
        json.dump(fimbs, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="path to parent of IQON3000.")
    parser.add_argument("--min_set_size", type=int, default=4, help="minimun set size (int).")
    parser.add_argument("--n_candidates", type=int, default=4, help="the number of fitb candidates (int).")
    parser.add_argument("--n_mix", type=int, default=4, help="mixed number of sets (int).")
    parser.add_argument("--max_set_size_x", type=int, default=8, help="maximum query set size (int).")
    parser.add_argument("--max_set_size_y", type=int, default=2, help="maximum candidate set size (int).")
    parser.add_argument("--train_size", type=float, default=0.8, help="training ratio (float).")
    parser.add_argument("--test_size", type=float, default=0.1, help="validation ratio (float).")
    args = parser.parse_args()
    main(args)
