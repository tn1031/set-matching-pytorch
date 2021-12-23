import numpy as np
import torch
from set_matching.losses import chamfer_loss, kl_loss
from sklearn.neighbors import NearestNeighbors


def np_chamfer_distance(x, y, metric="l2"):
    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric).fit(x)
    min_y_to_x = x_nn.kneighbors(y)[0]
    y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric).fit(y)
    min_x_to_y = y_nn.kneighbors(x)[0]
    chamfer_dist = np.sum(min_y_to_x ** 2) + np.sum(min_x_to_y ** 2)
    return chamfer_dist


def test_chamfer_loss():
    batchsize, n_units, sentence_length = 3, 32, 6
    x = torch.rand(batchsize, n_units, sentence_length)
    y = torch.rand(batchsize, n_units, sentence_length)
    mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        ]
    )

    loss = chamfer_loss(x, y, mask)

    dist = 0
    x = x.numpy().transpose(0, 2, 1)
    y = y.numpy().transpose(0, 2, 1)
    for _x, _y, _m in zip(x, y, mask):
        dist += np_chamfer_distance(_x[_m, :], _y[_m, :])

    assert np.isclose(loss.numpy().item(), dist / batchsize)


def test_kl_loss():
    batchsize, n_units, sentence_length = 3, 32, 6
    x = torch.rand(batchsize, n_units, sentence_length)
    y = torch.rand(batchsize, n_units, sentence_length)
    dx = torch.rand(batchsize, n_units, sentence_length)
    dy = torch.rand(batchsize, n_units, sentence_length)

    loss = kl_loss(x, y, dx, dy)
    assert loss.shape[0] == batchsize
