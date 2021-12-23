import torch
from set_matching.models.modules import make_attn_mask
from set_matching.models.set_vae import DecoderBlock, EncoderBlock, GaussianMixture, SetVAE


def test_gaussian_mixture():
    dim_z0, dim_output, n_mixtures = 4, 16, 3
    m = GaussianMixture(dim_z0, dim_output, n_mixtures, is_train=True)
    m.eval()

    mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        ]
    )
    z = m(mask)
    assert z.shape == (3, dim_output, 6)

    m = GaussianMixture(dim_z0, dim_output, n_mixtures, is_train=False)
    m.eval()
    z = m(mask)
    assert z.shape == (3, dim_output, 6)


def test_encoder_block():
    n_units, dim_i = 128, 16
    m = EncoderBlock(n_units, n_heads=8, m=dim_i)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    i_mask = torch.tensor([[True] * dim_i] * 2)
    xi_mask = make_attn_mask(x_mask, i_mask)
    ix_mask = make_attn_mask(i_mask, x_mask)
    y, h = m(x, ix_mask, xi_mask)
    assert y.shape == (batchsize, n_units, sentence_length)
    assert h.shape == (batchsize, n_units, dim_i)

    # permutation equivariant
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]
    y_perm, h_perm = m(x_perm, ix_mask, xi_mask)
    assert torch.all(torch.isclose(y[:, :, [1, 2, 3, 0, 4, 5, 6, 7]], y_perm, atol=1e-6))
    # permutation invariant
    assert torch.all(torch.isclose(h, h_perm, atol=1e-6))


def test_decoder_block():
    n_units, dim_i = 128, 16
    m = DecoderBlock(n_units, n_heads=8, m=dim_i)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    i_mask = torch.tensor([[True] * dim_i] * 2)
    xi_mask = make_attn_mask(x_mask, i_mask)
    ix_mask = make_attn_mask(i_mask, x_mask)
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]

    # project
    h = m.first_mab(x, ix_mask)
    h_perm = m.first_mab(x_perm, ix_mask)
    assert h.shape == (batchsize, n_units, dim_i)
    assert torch.all(torch.isclose(h, h_perm, atol=1e-6))  # permutation invariant

    # compute_prior
    z, mu, ln_var = m.compute_prior(h)
    assert z.shape == (batchsize, n_units, dim_i)
    assert mu.shape == (batchsize, n_units, dim_i)
    assert ln_var.shape == (batchsize, n_units, dim_i)

    # compute_posterior
    z, delta_mu, delta_ln_var = m.compute_posterior(mu, ln_var, h, h)
    assert z.shape == (batchsize, n_units, dim_i)
    assert delta_mu.shape == (batchsize, n_units, dim_i)
    assert delta_ln_var.shape == (batchsize, n_units, dim_i)

    # broadcast
    y = m.second_mab(x, z, xi_mask)
    y_perm = m.second_mab(x_perm, z, xi_mask)
    assert y.shape == (batchsize, n_units, sentence_length)
    assert torch.all(torch.isclose(y[:, :, [1, 2, 3, 0, 4, 5, 6, 7]], y_perm, atol=1e-6))


def test_set_vae():
    n_units = 64
    n_heads = 8
    z_length = [2, 4, 8]
    dim_z0 = 16
    n_mixtures = 16
    m = SetVAE(n_units, n_heads, z_length, dim_z0, n_mixtures)
    m.eval()

    batchsize, sentence_length = 2, 8
    x = torch.rand(batchsize, n_units, sentence_length)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    x_perm = x[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]

    torch.manual_seed(0)
    y, params = m(x, x_mask)
    torch.manual_seed(0)
    y_perm, params_perm = m(x_perm, x_mask)
    assert y.shape == (batchsize, n_units, sentence_length)
    assert torch.all(torch.isclose(y, y_perm, atol=1e-6))
    for p, pp, l in zip(params, params_perm, z_length):
        for pl, ppl in zip(p, pp):
            assert pl.shape == (batchsize, n_units, l)
            assert torch.all(torch.isclose(pl, ppl, atol=1e-6))

    h_enc = m.encode(x, x_mask)
    z0 = m.sample_z0(x_mask)
    z0_perm = z0[:, :, [1, 2, 3, 0, 4, 5, 6, 7]]
    torch.manual_seed(0)
    y, params = m.decode(z0, x_mask, list(reversed(h_enc)))
    torch.manual_seed(0)
    y_perm, params_perm = m.decode(z0_perm, x_mask, list(reversed(h_enc)))
    assert torch.all(torch.isclose(y[:, :, [1, 2, 3, 0, 4, 5, 6, 7]], y_perm, atol=1e-6))
    for p, pp, l in zip(params, params_perm, z_length):
        for pl, ppl in zip(p, pp):
            assert pl.shape == (batchsize, n_units, l)
            assert torch.all(torch.isclose(pl, ppl, atol=1e-6))


if __name__ == "__main__":
    test_set_vae()
