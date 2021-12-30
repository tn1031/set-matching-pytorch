import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from set_matching.models.modules import ISAB, FeedForwardLayer, make_attn_mask

PADDING_IDX = 0

# An implementation of SetVAE
# https://arxiv.org/abs/2103.15619
# Some modules are based on the following.
# https://github.com/jw9730/setvae


class GaussianMixture(nn.Module):
    def __init__(self, dim_z0, dim_output, n_mixtures, is_train):
        super().__init__()
        self.dim_z0 = dim_z0
        self.n_mixtures = n_mixtures
        self.is_train = is_train
        self.tau = 1.0
        register = self.register_parameter if is_train else self.register_buffer

        if n_mixtures == 1:
            register("mu", nn.Parameter(torch.randn(1, 1, dim_z0)))
            register("ln_var", nn.Parameter(torch.randn(1, 1, dim_z0)))
            nn.init.xavier_uniform_(self.mu)
            nn.init.xavier_uniform_(self.ln_var)
        else:
            register("logits", nn.Parameter(torch.ones(n_mixtures)))
            register("mu", nn.Parameter(torch.randn(n_mixtures, dim_z0)))
            register("ln_var", nn.Parameter(torch.randn(n_mixtures, dim_z0)))
            nn.init.xavier_uniform_(self.mu)
            nn.init.xavier_uniform_(self.ln_var)
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
                component_id_onehot = F.gumbel_softmax(logits, tau=self.tau, hard=True).unsqueeze(-1)
                # assert onehot.shape == (batchsize, max_length, self.n_mixtures, 1)

                mu = self.mu.reshape([1, 1, self.n_mixtures, self.dim_z0])
                ln_var = self.ln_var.reshape([1, 1, self.n_mixtures, self.dim_z0])
                mu = (mu * component_id_onehot).sum(dim=2)
                ln_var = (ln_var * component_id_onehot).sum(dim=2)
                z0 = mu + ln_var.exp() * eps
            else:
                mix = D.Categorical(self.logits)
                component = D.Independent(D.Normal(self.mu, self.ln_var.exp()), 1)
                mixture = D.MixtureSameFamily(mix, component)
                z0 = mixture.sample((batchsize, max_length))

        # assert z0.shape == (batchsize, max_length, self.dim_z0)
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
        _, n_units, _ = h_enc.shape
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
        n_heads=8,
        z_length=[2, 4, 8],
        dim_z0=16,
        n_mixtures=16,
    ):
        super(SetVAE, self).__init__()
        self.n_units = n_units
        self.n_layers = len(z_length)
        self.z_length = z_length
        self.gmm = GaussianMixture(dim_z0, n_units, n_mixtures, False)

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

    def decode(self, z, z_mask, h_enc_list, intermediate_features=False):
        batchsize = len(z)
        if intermediate_features:
            features = {"h_enc": [], "h_dec": [], "z": []}

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

            if intermediate_features:
                features["h_enc"].append(h_enc)
                features["h_dec"].append(h)
                features["z"].append(z)

        if intermediate_features:
            return z_prev, z_params, features
        else:
            return z_prev, z_params

    def forward(self, x, x_mask):
        h_enc = self.encode(x, x_mask)
        # for h, l in zip(h_enc, list(reversed(self.z_length))):
        #    assert h.shape == (batch_size, n_units, l)
        z0 = self.sample_z0(x_mask)
        # assert z0.shape == (batch_size, n_units, len_x)
        x_hat, params = self.decode(z0, x_mask, list(reversed(h_enc)))
        # assert x_hat.shape == (batch_size, n_units, len_x)
        return x_hat, params

    def intermediate_features(self, x, x_mask):
        h_enc = self.encode(x, x_mask)
        z0 = self.sample_z0(x_mask)
        x_hat, params, features = self.decode(z0, x_mask, list(reversed(h_enc)), intermediate_features=True)
        features["z0"] = z0
        prior_mu, prior_ln_var, posterior_mu, posterior_ln_var = zip(*params)
        features["prior_mu"] = prior_mu
        features["prior_ln_var"] = prior_ln_var
        features["posterior_mu"] = posterior_mu
        features["posterior_ln_var"] = posterior_ln_var
        return x_hat, features
