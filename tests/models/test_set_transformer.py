import torch

from set_matching.models.set_transformer import SetTransformer


def test_set_transformer():
    n_units, n_encoder_layers, n_heads, n_output_instances = 64, 2, 8, 1
    m = SetTransformer(
        n_units, n_encoder_layers=n_encoder_layers, n_heads=n_heads, n_output_instances=n_output_instances
    )
    m.eval()

    batchsize, sentence_length = 3, 8
    x = torch.rand(batchsize, sentence_length, 3, 244, 244)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4, [True] * 3 + [False] * 5])
    t = torch.rand(batchsize, 3, 244, 244)
    y = m(x, x_mask, t)
    assert y.shape == (batchsize, batchsize)
