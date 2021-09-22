import torch
from set_matching.models.set_matching import SetMatching


def test_set_matching_image():
    n_units, n_encoder_layer, n_decoder_layer, n_heads, n_iterative = 64, 2, 2, 8, 2
    m = SetMatching(
        n_units,
        n_encoder_layer=n_encoder_layer,
        n_decoder_layer=n_decoder_layer,
        n_heads=n_heads,
        n_iterative=n_iterative,
    )
    m.eval()

    batchsize, sentence_length = 3, 8
    x = torch.rand(batchsize, sentence_length, 3, 244, 244)
    y = torch.rand(batchsize, sentence_length + 1, 3, 244, 244)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4, [True] * 3 + [False] * 5])
    y_mask = torch.tensor([[True] * 4 + [False] * 5, [True] * 5 + [False] * 4, [True] * 3 + [False] * 6])

    x_y = m(x, x_mask, y, y_mask)
    assert x_y.shape == (batchsize, batchsize)
    y_x = m(y, y_mask, x, x_mask)
    assert y_x.shape == (batchsize, batchsize)

    # permutation invariant
    x_perm = x[:, [1, 0, 2, 3, 4, 5, 6, 7], ...]
    x_perm_y = m(x_perm, x_mask, y, y_mask)
    assert torch.all(torch.isclose(x_y, x_perm_y, atol=1e-6))
    # permutation invariant
    y_perm = y[:, [1, 0, 2, 3, 4, 5, 6, 7, 8], ...]
    x_y_perm = m(x, x_mask, y_perm, y_mask)
    assert torch.all(torch.isclose(x_y, x_y_perm, atol=1e-6))


def test_set_matching_feature():
    n_units, n_encoder_layer, n_decoder_layer, n_heads, n_iterative = 64, 2, 2, 8, 2
    m = SetMatching(
        n_units,
        n_encoder_layer=n_encoder_layer,
        n_decoder_layer=n_decoder_layer,
        n_heads=n_heads,
        n_iterative=n_iterative,
        embedder_arch="linear",
    )
    m.eval()

    batchsize, sentence_length = 3, 8
    x = torch.rand(batchsize, sentence_length, 4096)
    y = torch.rand(batchsize, sentence_length + 1, 4096)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4, [True] * 3 + [False] * 5])
    y_mask = torch.tensor([[True] * 4 + [False] * 5, [True] * 5 + [False] * 4, [True] * 3 + [False] * 6])

    x_y = m(x, x_mask, y, y_mask)
    assert x_y.shape == (batchsize, batchsize)
    y_x = m(y, y_mask, x, x_mask)
    assert y_x.shape == (batchsize, batchsize)

    # permutation invariant
    x_perm = x[:, [1, 0, 2, 3, 4, 5, 6, 7], ...]
    x_perm_y = m(x_perm, x_mask, y, y_mask)
    assert torch.all(torch.isclose(x_y, x_perm_y, atol=1e-6))
    # permutation invariant
    y_perm = y[:, [1, 0, 2, 3, 4, 5, 6, 7, 8], ...]
    x_y_perm = m(x, x_mask, y_perm, y_mask)
    assert torch.all(torch.isclose(x_y, x_y_perm, atol=1e-6))
