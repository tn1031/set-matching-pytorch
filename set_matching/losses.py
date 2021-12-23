import torch.nn as nn


def kl_loss(mu, ln_var, delta_mu, delta_ln_var):
    batch_size = mu.shape[0]
    loss = -0.5 * (delta_ln_var + 1.0 - delta_mu.pow(2) / ln_var.exp().pow(2) - delta_ln_var.exp().pow(2))
    loss = loss.view(batch_size, -1).sum(dim=-1)
    return loss


def chamfer_loss(x, y, mask):
    cardinality = mask.sum(dim=1).tolist()
    mask = mask.flatten()
    x = x.permute(0, 2, 1).flatten(0, 1)[mask, :]
    y = y.permute(0, 2, 1).flatten(0, 1)[mask, :]

    x = x.split(cardinality, 0)
    y = y.split(cardinality, 0)
    dist = []
    for _x, _y in zip(x, y):
        len_x, len_y = _x.size(0), _y.size(0)
        _x = _x.unsqueeze(1).repeat(1, len_y, 1)
        _y = _y.unsqueeze(0).repeat(len_x, 1, 1)
        l2 = (_x - _y).pow(2).sum(dim=-1)
        x_dist = l2.min(dim=0)[0].sum()
        y_dist = l2.min(dim=1)[0].sum()
        dist.append(x_dist + y_dist)
    loss = sum(dist) / len(dist)
    return loss


class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, mask):
        return chamfer_loss(x, y, mask)


class HierarchicalKLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, params):
        loss = 0
        for param in params:
            loss += kl_loss(*param)
        return loss.mean()
