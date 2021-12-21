from math import sqrt

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from set_matching.models.cnn import CNN
from set_matching.models.helper import get_mixture_parameters
from set_matching.models.modules import ISAB, FeedForwardLayer, MultiHeadAttention, make_attn_mask

PADDING_IDX = 0


class GaussianMixture(nn.Module):
    def __init__(self, dim_z0, dim_output, n_mixtures, is_train):
        super().__init__()
        self.dim_z0 = dim_z0
        self.n_mixtures = n_mixtures
        self.is_train = is_train
        self.tau = 1.0
        self.register = self.register_parameter if is_train else self.register_buffer

        fixed_gmm = False
        if n_mixtures == 1:
            self.register("mu", nn.Parameter(torch.randn(1, 1, dim_z0)))  # [1, 1, Ds]
            self.register("ln_var", nn.Parameter(torch.randn(1, 1, dim_z0)))  # [1, 1, Ds]
            nn.init.xavier_uniform_(self.mu)
            nn.init.xavier_uniform_(self.ln_var)
        elif fixed_gmm:
            logits, mu, sig = get_mixture_parameters(n_mixtures, dim_z0)
            self.register("logits", nn.Parameter(logits))
            self.register("mu", nn.Parameter(mu))
            self.register("sigma", nn.Parameter(sig))
        else:
            self.register("logits", nn.Parameter(torch.ones(n_mixtures)))
            self.register("mu", nn.Parameter(torch.randn(n_mixtures, dim_z0)))
            self.register("sigma", nn.Parameter(torch.randn(n_mixtures, dim_z0).abs() / sqrt(n_mixtures)))
        self.output_layer = nn.Linear(dim_z0, dim_output)

    def forward(self, mask):
        """
        z in R^{n \times d} ~ N(mu_0, sigma_0)
        h = fc(z) in R^{n \times d}
        """
        batchsize, max_length = mask.shape
        eps = torch.randn([batchsize, max_length, self.dim_z0]).to(mask.device)

        if self.n_mixtures == 1:
            z0 = self.mu + self.ln_var.exp() * eps
        else:
            if self.is_train:
                logits = self.logits.reshape([1, 1, self.n_mixtures]).repeat(batchsize, max_length, 1)
                onehot = F.gumbel_softmax(logits, tau=self.tau, hard=True).unsqueeze(-1)
                assert onehot.shape == (batchsize, max_length, self.n_mixtures, 1)

                mu = self.mu.reshape([1, 1, self.n_mixtures, self.dim_z0])
                sigma = self.sigma.reshape([1, 1, self.n_mixtures, self.dim_z0])
                mu = (mu * onehot).sum(dim=2)
                sigma = (sigma * onehot).sum(dim=2)
                z0 = mu + sigma * eps
            else:
                mix = D.Categorical(self.logits)
                comp = D.Independent(D.Normal(self.mu, self.sigma), 1)
                mixture = D.MixtureSameFamily(mix, comp)
                z0 = mixture.sample((batchsize, max_length))

        assert z0.shape == (batchsize, max_length, self.dim_z0)
        return self.output_layer(z0).permute(0, 2, 1)


class EncoderBlock(ISAB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, ix_mask, xi_mask):
        """
        ISAB(X) = MAB(X, H) in R^{n \times d}
        where H = MAB(I, X) in R^{m \times d}
        MAB(u, v) = LayerNorm(H + rFF(H))
        where H = LayerNorm(u + Multihead(u, v, v))
        """
        batch, n_units, _ = x.shape
        # MAB(I, X), mask: m -> n
        i = torch.broadcast_to(self.paramI, (batch, n_units, self.m))
        mha_ix = self.self_attention_1(i, x, mask=ix_mask)
        h = self.ln_1_1(i + mha_ix)
        rff = self.feed_forward_1(h)
        mab_ix = self.ln_1_2(h + rff)
        # MAB(X, H), mask: n -> m
        mha_xh = self.self_attention_2(x, mab_ix, mask=xi_mask)
        h = self.ln_2_1(x + mha_xh)
        rff = self.feed_forward_2(h)
        h = self.ln_2_2(h + rff)
        return h, mab_ix


class DecoderBlock(ISAB):
    def __init__(self, n_units, n_heads=8, m=16, cond_prior=True):
        super().__init__(n_units, n_heads, m)
        if cond_prior:
            self.prior = FeedForwardLayer(n_units, 2 * n_units)
        else:
            self.register_parameter(name="prior", param=nn.Parameter(torch.randn(1, m, 2 * n_units)))
            nn.init.xavier_uniform_(self.prior)
        self.posterior = FeedForwardLayer(n_units, 2 * n_units)
        self.ff = FeedForwardLayer(n_units, n_units)

        self.cond_prior = cond_prior

    def first_mab(self, x, ix_mask):
        batch, n_units, _ = x.shape
        # MAB(I, X), mask: m -> n
        i = torch.broadcast_to(self.paramI, (batch, n_units, self.m))
        mha_ix = self.self_attention_1(i, x, mask=ix_mask)
        h = self.ln_1_1(i + mha_ix)
        rff = self.feed_forward_1(h)
        mab_ix = self.ln_1_2(h + rff)
        return mab_ix

    def second_mab(self, x, z, xi_mask):
        mha_xh = self.self_attention_2(x, self.ff(z), mask=xi_mask)
        h = self.ln_2_1(x + mha_xh)
        rff = self.feed_forward_2(h)
        h = self.ln_2_2(h + rff)
        return h

    def compute_prior(self, h):
        batch, n_units, _ = h.shape
        chunk_size = n_units
        if self.cond_prior:
            prior = self.prior(h)
        else:
            prior = self.prior.repeat(batch, 1, 1)
        mu, ln_var = torch.split(prior, chunk_size, dim=1)
        # ln_var = ln_var.clamp(-2., 1.5)
        eps = torch.randn(mu.shape).to(h)
        z = mu + ln_var.exp() * eps
        return z, mu, ln_var

    def compute_posterior(self, mu, ln_var, h_enc, h_abl=None):
        # eq. (20)
        batch, n_units, len_z = h_enc.shape
        chunk_size = n_units
        h = h_enc + h_abl if h_abl is not None else h_enc
        posterior = self.posterior(h)
        delta_mu, delta_ln_var = torch.split(posterior, chunk_size, dim=1)
        eps = torch.randn(mu.shape).to(mu)
        z = (mu + delta_mu) + (ln_var.exp() * delta_ln_var.exp()) * eps
        return z, delta_mu, delta_ln_var


class SetVAE(nn.Module):
    def __init__(
        self,
        n_units,
        dim_hidden=128,
        n_heads=8,
        z_length=[2, 4, 8],
        dim_z0=16,
        n_mixtures=16,
        embedder_arch="resnet18",
        disable_cnn_update=False,
    ):
        super(SetVAE, self).__init__()
        self.n_units = n_units
        self.n_layers = len(z_length)
        self.z_length = z_length
        # self.input_layer = nn.Linear(dim_input, dim_hidden)
        self.gmm = GaussianMixture(dim_z0, n_units, n_mixtures, False)
        # self.pre_encoder = FeedForwardLayer(dim_hidden)
        # self.pre_decoder = FeedForwardLayer(dim_hidden)

        enc_len_x = list(reversed(z_length))
        dec_len_x = z_length
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(EncoderBlock(n_units, n_heads=n_heads, m=enc_len_x[i]))
            self.register_buffer(f"i_mask_{i}", torch.ones((1, enc_len_x[i])).bool())
            self.decoder.append(DecoderBlock(n_units, n_heads=n_heads, m=dec_len_x[i]))

    def encode(self, x, x_mask):
        batchsize = len(x)

        h_enc = []
        for i, layer in enumerate(self.encoder):
            i_mask = getattr(self, f"i_mask_{i}").repeat(batchsize, 1)
            xi_mask = make_attn_mask(x_mask, i_mask)
            ix_mask = make_attn_mask(i_mask, x_mask)
            x, h = layer(x, ix_mask, xi_mask)
            h_enc.append(h)
        return h_enc

    def sample_z0(self, mask):
        z0 = self.gmm(mask)
        return z0

    def decode(self, z, z_mask, h_enc_list):
        batchsize = len(z)

        z_params = []
        z_prev = z
        for i, (layer, h_enc) in enumerate(zip(self.decoder, h_enc_list)):
            i_mask = getattr(self, f"i_mask_{self.n_layers - i - 1}").repeat(batchsize, 1)
            iz_mask = make_attn_mask(i_mask, z_mask)
            zi_mask = make_attn_mask(z_mask, i_mask)
            h = layer.first_mab(z_prev, iz_mask)  # MAB(I, z_prev)

            _, mu, ln_var = layer.compute_prior(h)
            z, delta_mu, delta_ln_var = layer.compute_posterior(mu, ln_var, h_enc, None if i == 0 else h)  # z ~ N
            # assert z.shape == (batchsize, self.n_units, self.z_length[i])
            z_next = layer.second_mab(z_prev, z, zi_mask)

            z_params.append((mu, ln_var, delta_mu, delta_ln_var))
            z_prev = z_next

        return z_prev, z_params

    def forward(self, x, x_mask):
        batch_size, n_units, len_x = x.shape
        # input layer
        # pre encoder
        h_enc = self.encode(x, x_mask)
        # for h, l in zip(h_enc, list(reversed(self.z_length))):
        #    assert h.shape == (batch_size, n_units, l)
        z0 = self.sample_z0(x_mask)
        # assert z0.shape == (batch_size, n_units, len_x)
        # pre decoder
        x_hat, params = self.decode(z0, x_mask, list(reversed(h_enc)))
        # assert x_hat.shape == (batch_size, n_units, len_x)
        # post decoder
        # output layer
        # .postprocess
        return x_hat, params
