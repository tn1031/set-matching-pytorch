import torch
from set_matching.models.modules import (
    ISAB,
    MAB,
    PMA,
    SAB,
    ConvolutionSentence,
    CrossSetDecoder,
    FeedForwardLayer,
    LayerNormalizationSentence,
    MultiHeadAttention,
    MultiHeadExpectation,
    MultiHeadSimilarity,
    SetDecoder,
    SetEncoder,
    SetISABEncoder,
    SlotAttention,
    StackedCrossSetDecoder,
    make_attn_mask,
)


def test_make_attn_mask():
    """
              target
              T F F F
             ---------
           T| T F F F
    source T| T F F F
           F| F F F F
    """
    mask_x = torch.tensor([[True, True, False]])
    mask_y = torch.tensor([[True, False, False, False]])

    mask_xx = torch.tensor([[True, True, False], [True, True, False], [False, False, False]])
    mask_xy = torch.tensor([[True, False, False, False], [True, False, False, False], [False, False, False, False]])

    assert torch.all(make_attn_mask(mask_x, mask_x) == mask_xx)
    assert torch.all(make_attn_mask(mask_x, mask_y) == mask_xy)


def test_convolution_sentence():
    in_channels, out_channels = 32, 64
    m = ConvolutionSentence(in_channels, out_channels, bias=False)
    torch.nn.init.ones_(m.weight)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, in_channels, sentence_length)
    y = m(x)
    assert y.shape == (batchsize, out_channels, sentence_length)
    assert torch.all(torch.isclose(y[0, 0, :], x[0, :, :].sum(0)))


def test_layer_normalization_sentence():
    n_units = 64
    m = LayerNormalizationSentence(n_units, eps=1e-6)
    torch.nn.init.ones_(m.weight)
    m.reset_parameters()
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    y = m(x)
    assert y.shape == (batchsize, n_units, sentence_length)


def test_feed_forward_layer():
    n_units = 64
    m = FeedForwardLayer(n_units)
    torch.nn.init.ones_(m.w_1.weight)
    torch.nn.init.zeros_(m.w_1.bias)
    torch.nn.init.ones_(m.w_2.weight)
    torch.nn.init.zeros_(m.w_2.bias)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = -torch.abs(torch.rand(batchsize, n_units, sentence_length))
    y = m(x)
    assert y.shape == (batchsize, n_units, sentence_length)
    _x = torch.cat([x.sum(dim=1, keepdim=True)] * n_units * 4, dim=1)
    _x = torch.where(_x < 0, 0.2 * _x, _x)
    _y = torch.cat([_x.sum(dim=1, keepdim=True)] * n_units, dim=1)
    assert torch.all(torch.isclose(y, _y, atol=1e-6))


def test_multiead_softmax_self_attention():
    n_units = 128
    m = MultiHeadAttention(n_units, n_heads=8, self_attention=True, activation_fn="softmax")
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    xx_mask = make_attn_mask(mask, mask)
    y = m(x, mask=xx_mask)
    assert y.shape == (batchsize, n_units, sentence_length)
    assert torch.all(y[0, :, -3:] == torch.zeros((n_units, 3), dtype=torch.float32))
    assert torch.all(y[1, :, -4:] == torch.zeros((n_units, 4), dtype=torch.float32))

    # permutation equivariant
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]
    y_perm = m(x_perm, mask=xx_mask)
    assert torch.all(torch.isclose(y[:, :, [1, 2, 3, 0, 4, 5, 6, 7]], y_perm, atol=1e-6))


def test_multiead_softmax_attention():
    n_units = 128
    m = MultiHeadAttention(
        n_units, n_heads=8, self_attention=False, activation_fn="softmax", normalize_attn=True, finishing_linear=False
    )
    m.eval()

    batchsize, sentence_length, query_length = 2, 8, 4
    x = torch.rand(batchsize, n_units, sentence_length)
    y = torch.rand(batchsize, n_units, query_length)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    y_mask = torch.tensor([[True] * 3 + [False] * 1, [True] * 2 + [False] * 2])
    yx_mask = make_attn_mask(y_mask, x_mask)
    z = m(y, x, mask=yx_mask)
    assert z.shape == (batchsize, n_units, query_length)
    # assert torch.all(z[0, :, -1:] == torch.zeros((n_units, 1), dtype=torch.float32))
    # assert torch.all(z[1, :, -2:] == torch.zeros((n_units, 2), dtype=torch.float32))
    assert torch.all(torch.isnan(z[0, :, -1:]))
    assert torch.all(torch.isnan(z[1, :, -2:]))

    # permutation invariant
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]
    z_perm = m(y, x_perm, mask=yx_mask)
    assert torch.all(torch.isclose(z[0, :, :3], z_perm[0, :, :3], atol=1e-6))
    assert torch.all(torch.isclose(z[1, :, :2], z_perm[1, :, :2], atol=1e-6))

    # permutation equivariant
    y_perm = y[:, :, [1, 0, 2, 3]]
    z_perm = m(y_perm, x, mask=yx_mask)
    assert torch.all(torch.isclose(z[0, :, [1, 0, 2]], z_perm[0, :, :3], atol=1e-6))
    assert torch.all(torch.isclose(z[1, :, [1, 0]], z_perm[1, :, :2], atol=1e-6))


def test_multiead_relu_attention():
    n_units = 128
    m = MultiHeadAttention(n_units, n_heads=8, self_attention=True, activation_fn="relu")
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    xx_mask = make_attn_mask(mask, mask)
    y = m(x, mask=xx_mask)
    assert y.shape == (batchsize, n_units, sentence_length)
    assert torch.all(y[0, :, -3:] == torch.zeros((n_units, 3), dtype=torch.float32))
    assert torch.all(y[1, :, -4:] == torch.zeros((n_units, 4), dtype=torch.float32))

    # permutation equivariant
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]
    y_perm = m(x_perm, mask=xx_mask)
    assert torch.all(torch.isclose(y[:, :, [1, 2, 3, 0, 4, 5, 6, 7]], y_perm, atol=1e-6))


def test_multihead_similarity():
    n_units, n_heads = 128, 8
    m = MultiHeadSimilarity(n_units, n_heads)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    y = torch.rand(batchsize, n_units, sentence_length + 1)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    y_mask = torch.tensor([[True] * 4 + [False] * 5, [True] * 5 + [False] * 4])
    xy_mask = make_attn_mask(x_mask, y_mask)
    yx_mask = make_attn_mask(y_mask, x_mask)
    # x_y = m(x * torch.cat([x_mask[None, :, :]] * n_units, dim=0).permute(1, 0, 2), y, xy_mask=xy_mask)
    x_y = m(x, y, xy_mask)
    assert x_y.shape == (batchsize, n_units, sentence_length)
    y_x = m(y, x, yx_mask)
    assert y_x.shape == (batchsize, n_units, sentence_length + 1)

    # permutation equivariant
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]
    x_perm_y = m(x_perm, y, xy_mask)
    assert torch.all(torch.isclose(x_y[:, :, [1, 2, 3, 0, 4, 5, 6, 7]], x_perm_y, atol=1e-6))
    # permutation invariant
    y_perm = y[:, :, [1, 0, 2, 3, 4, 5, 6, 7, 8]]
    x_y_perm = m(x, y_perm, xy_mask)
    assert torch.all(torch.isclose(x_y, x_y_perm, atol=1e-6))

    attn = m.get_attnmap(x, y, xy_mask)
    assert attn.shape == (batchsize * n_heads, sentence_length, sentence_length + 1)


def test_multihead_expectation():
    n_units, n_heads = 128, 8
    m = MultiHeadExpectation(n_units, n_heads)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    y = torch.rand(batchsize, n_units, sentence_length + 1)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    y_mask = torch.tensor([[True] * 4 + [False] * 5, [True] * 5 + [False] * 4])
    xy_mask = make_attn_mask(x_mask, y_mask)
    yx_mask = make_attn_mask(y_mask, x_mask)
    x_y = m(x, y, xy_mask)
    assert x_y.shape == (batchsize, 1)
    y_x = m(y, x, yx_mask)
    assert y_x.shape == (batchsize, 1)

    # permutation invariant
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]
    x_perm_y = m(x_perm, y, xy_mask)
    assert torch.all(torch.isclose(x_y, x_perm_y, atol=1e-6))
    # permutation invariant
    y_perm = y[:, :, [1, 0, 2, 3, 4, 5, 6, 7, 8]]
    x_y_perm = m(x, y_perm, xy_mask)
    assert torch.all(torch.isclose(x_y, x_y_perm, atol=1e-6))


def test_sab():
    n_units = 128
    m = SAB(n_units, n_heads=8)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    xx_mask = make_attn_mask(mask, mask)
    y = m(x, xx_mask)
    assert y.shape == (batchsize, n_units, sentence_length)

    # permutation equivariant
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]
    y_perm = m(x_perm, xx_mask)
    assert torch.all(torch.isclose(y[:, :, [1, 2, 3, 0, 4, 5, 6, 7]], y_perm, atol=1e-6))


def test_isab():
    n_units, dim_i = 128, 16
    m = ISAB(n_units, n_heads=8, m=dim_i)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    i_mask = torch.tensor([[True] * dim_i] * 2)
    xi_mask = make_attn_mask(x_mask, i_mask)
    ix_mask = make_attn_mask(i_mask, x_mask)
    y = m(x, ix_mask, xi_mask)
    assert y.shape == (batchsize, n_units, sentence_length)

    # permutation equivariant
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]
    y_perm = m(x_perm, ix_mask, xi_mask)
    assert torch.all(torch.isclose(y[:, :, [1, 2, 3, 0, 4, 5, 6, 7]], y_perm, atol=1e-6))


def test_mab():
    n_units = 128
    m = MAB(n_units, n_heads=8)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    z = torch.rand(batchsize, n_units, sentence_length + 1)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    z_mask = torch.tensor([[True] * 5 + [False] * 4, [True] * 4 + [False] * 5])
    xz_mask = make_attn_mask(x_mask, z_mask)
    y = m(x, z, xz_mask)
    assert y.shape == (batchsize, n_units, sentence_length)

    # permutation equivariant
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]
    y_perm = m(x_perm, z, xz_mask)
    assert torch.all(torch.isclose(y[:, :, [1, 2, 3, 0, 4, 5, 6, 7]], y_perm, atol=1e-6))


def test_pma():
    n_units, n_output_instances = 128, 2
    m = PMA(n_units, n_heads=8, n_output_instances=n_output_instances)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)

    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    s_mask = torch.tensor([[True, True], [True, True]])
    sx_mask = make_attn_mask(s_mask, x_mask)
    y = m(x, sx_mask)
    assert y.shape == (batchsize, n_units, n_output_instances)

    # permutation invariant
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]
    y_perm = m(x_perm, sx_mask)
    assert torch.all(torch.isclose(y, y_perm, atol=1e-6))

    n_output_instances = 2
    m = PMA(n_units, n_heads=8, n_output_instances=None)
    m.eval()

    s = torch.rand(n_output_instances, n_units)
    y = m(x, sx_mask, s=s)
    assert y.shape == (batchsize, n_units, n_output_instances)

    # permutation invariant
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]
    y_perm = m(x_perm, sx_mask, s=s)
    assert torch.all(torch.isclose(y, y_perm, atol=1e-6))

    # permutation equivariant
    s_perm = s[[1, 0], :]
    y_perm = m(x_perm, sx_mask, s=s_perm)
    assert torch.all(torch.isclose(y[:, :, [1, 0]], y_perm, atol=1e-6))  # (batch, n_units, n_output_instances)


def test_set_encoder():
    n_units, n_heads = 128, 8
    m = SetEncoder(n_units, n_heads=n_heads)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    xx_mask = make_attn_mask(mask, mask)
    y = m(x, xx_mask)
    assert y.shape == (batchsize, n_units, sentence_length)

    # permutation equivariant
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]
    y_perm = m(x_perm, xx_mask)
    assert torch.all(torch.isclose(y[:, :, [1, 2, 3, 0, 4, 5, 6, 7]], y_perm, atol=1e-6))

    attnmaps = m.get_attnmap(x, xx_mask)
    assert attnmaps[0].shape == (batchsize * n_heads, sentence_length, sentence_length)


def test_set_isab_encoder():
    n_units, n_heads, dim_i = 128, 8, 16
    m = SetISABEncoder(n_units, n_heads=n_heads, m=dim_i)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    i_mask = torch.tensor([[True] * dim_i] * 2)
    xi_mask = make_attn_mask(x_mask, i_mask)
    ix_mask = make_attn_mask(i_mask, x_mask)
    y = m(x, ix_mask, xi_mask)
    assert y.shape == (batchsize, n_units, sentence_length)

    # permutation equivariant
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]
    y_perm = m(x_perm, ix_mask, xi_mask)
    assert torch.all(torch.isclose(y[:, :, [1, 2, 3, 0, 4, 5, 6, 7]], y_perm, atol=1e-6))


def test_set_decoder():
    n_units, n_heads, n_output_instances = 128, 8, 2

    m = SetDecoder(n_units, n_heads, n_output_instances, apply_pma=True, apply_sab=True)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    y_mask = torch.tensor([[True, True], [True, True]])
    yx_mask = make_attn_mask(y_mask, x_mask)
    yy_mask = make_attn_mask(y_mask, y_mask)
    y = m(x, yx_mask, yy_mask)
    assert y.shape == (batchsize, n_units, n_output_instances)

    m = SetDecoder(n_units, n_heads, n_output_instances, apply_pma=False, apply_sab=True)
    m.eval()
    y = m(x, yx_mask, yy_mask)
    assert y.shape == (batchsize, n_units, n_output_instances)


def test_cross_set_decoder():
    n_units, n_heads = 128, 8
    m = CrossSetDecoder(n_units, n_heads, component="MHSim")
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    y = torch.rand(batchsize, n_units, sentence_length + 1)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    y_mask = torch.tensor([[True] * 4 + [False] * 5, [True] * 5 + [False] * 4])
    xy_mask = make_attn_mask(x_mask, y_mask)
    yx_mask = make_attn_mask(y_mask, x_mask)
    x_y = m(x, y, xy_mask)
    assert x_y.shape == (batchsize, n_units, sentence_length)
    y_x = m(y, x, yx_mask)
    assert y_x.shape == (batchsize, n_units, sentence_length + 1)

    # permutation equivariant
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]
    x_perm_y = m(x_perm, y, xy_mask)
    assert torch.all(torch.isclose(x_y[:, :, [1, 2, 3, 0, 4, 5, 6, 7]], x_perm_y, atol=1e-6))
    # permutation invariant
    y_perm = y[:, :, [1, 0, 2, 3, 4, 5, 6, 7, 8]]
    x_y_perm = m(x, y_perm, xy_mask)
    assert torch.all(torch.isclose(x_y, x_y_perm, atol=1e-6))

    m = CrossSetDecoder(n_units, n_heads, component="MAB")
    m.eval()

    x_y = m(x, y, xy_mask)
    assert x_y.shape == (batchsize, n_units, sentence_length)
    y_x = m(y, x, yx_mask)
    assert y_x.shape == (batchsize, n_units, sentence_length + 1)
    x_perm_y = m(x_perm, y, xy_mask)
    assert torch.all(torch.isclose(x_y[:, :, [1, 2, 3, 0, 4, 5, 6, 7]], x_perm_y, atol=1e-6))
    x_y_perm = m(x, y_perm, xy_mask)
    assert torch.all(torch.isclose(x_y, x_y_perm, atol=1e-6))

    m = CrossSetDecoder(n_units, n_heads, component="MHAtt")
    m.eval()

    x_y = m(x, y, xy_mask)
    assert x_y.shape == (batchsize, n_units, sentence_length)
    y_x = m(y, x, yx_mask)
    assert y_x.shape == (batchsize, n_units, sentence_length + 1)
    x_perm_y = m(x_perm, y, xy_mask)
    assert torch.all(torch.isclose(x_y[:, :, [1, 2, 3, 0, 4, 5, 6, 7]], x_perm_y, atol=1e-6))
    x_y_perm = m(x, y_perm, xy_mask)
    assert torch.all(torch.isclose(x_y, x_y_perm, atol=1e-6))


def test_stacked_cross_set_decoder():
    n_units, n_layers, n_heads = 128, 3, 8
    m = StackedCrossSetDecoder(n_units, n_layers, n_heads)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    y = torch.rand(batchsize, n_units, sentence_length + 1)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    y_mask = torch.tensor([[True] * 4 + [False] * 5, [True] * 5 + [False] * 4])
    xy_mask = make_attn_mask(x_mask, y_mask)
    yx_mask = make_attn_mask(y_mask, x_mask)
    x_y = m(x, y, xy_mask)
    assert x_y.shape == (batchsize, n_units, sentence_length)
    y_x = m(y, x, yx_mask)
    assert y_x.shape == (batchsize, n_units, sentence_length + 1)


def test_slot_attention():
    n_units, n_output_instances = 64, 4
    m = SlotAttention(n_units, n_output_instances=n_output_instances)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    y_mask = torch.tensor([[True] * 4, [True] * 3 + [False] * 1])
    yx_mask = make_attn_mask(y_mask, x_mask)
    x_y = m(x, yx_mask)
    assert x_y.shape == (batchsize, n_units, n_output_instances)
    assert not torch.any(torch.isnan(x_y[0, :, :]))
    assert torch.all(torch.isnan(x_y[1, :, -1]))

    y = torch.rand(batchsize, n_units, 6)
    y_mask = torch.tensor([[True] * 4 + [False] * 2, [True] * 3 + [False] * 3])
    yx_mask = make_attn_mask(y_mask, x_mask)
    x_y = m(x, yx_mask, slots=y)
    assert x_y.shape == (batchsize, n_units, 6)
    assert torch.all(torch.isnan(x_y[0, :, -2:]))
    assert torch.all(torch.isnan(x_y[1, :, -3:]))

    x_perm = x[:, :, [1, 0, 2, 3, 4, 5, 6, 7]]
    x_y_perm = m(x_perm, yx_mask, slots=y)
    assert torch.all(torch.isclose(x_y[0, :, :-2], x_y_perm[0, :, :-2], atol=1e-6))
    assert torch.all(torch.isclose(x_y[1, :, :-3], x_y_perm[1, :, :-3], atol=1e-6))

    y_perm = y[:, :, [1, 0, 2, 3, 4, 5]]
    x_y_perm = m(x, yx_mask, slots=y_perm)
    assert torch.all(torch.isclose(x_y[0, :, [1, 0, 2, 3]], x_y_perm[0, :, :-2], atol=1e-6))
    assert torch.all(torch.isclose(x_y[1, :, [1, 0, 2]], x_y_perm[1, :, :-3], atol=1e-6))
