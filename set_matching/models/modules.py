import torch
import torch.nn as nn

# Some modules are based on the following.
# https://github.com/soskek/attention_is_all_you_need


def make_attn_mask(source, target):
    mask = (target[:, None, :]) * (source[:, :, None])
    # (batch, source_length, target_length)
    return mask


def _seq_func(func, x, reconstruct_shape=True):
    batch, n_units, length = x.shape
    # transpose to move the channel to last dim.
    e = x.permute(0, 2, 1).reshape(batch * length, n_units)
    e = func(e)
    if not reconstruct_shape:
        return e
    out_units = e.shape[1]
    e = e.reshape((batch, length, out_units)).permute(0, 2, 1)
    # assert e.shape == (batch, out_units, length)
    return e


class LayerNormalizationSentence(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNormalizationSentence, self).__init__(*args, **kwargs)

    def __call__(self, x):
        y = _seq_func(super(LayerNormalizationSentence, self).forward, x)
        return y


class ConvolutionSentence(nn.Conv2d):
    def __init__(self, in_channels, out_channels, bias):
        super(ConvolutionSentence, self).__init__(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=3)
        y = super(ConvolutionSentence, self).forward(x)
        y = torch.squeeze(y, dim=3)
        return y


class FeedForwardLayer(nn.Module):
    def __init__(self, n_units):
        super(FeedForwardLayer, self).__init__()
        n_inner_units = n_units * 4
        self.w_1 = ConvolutionSentence(n_units, n_inner_units, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.w_2 = ConvolutionSentence(n_inner_units, n_units, bias=True)

    def __call__(self, e):
        e = self.w_1(e)
        e = self.leaky_relu(e)
        e = self.w_2(e)
        return e


class MultiHeadAttention(nn.Module):
    def __init__(self, n_units, n_heads=8, self_attention=True, activation_fn="relu"):
        super(MultiHeadAttention, self).__init__()
        # layers
        if self_attention:
            self.w_QKV = ConvolutionSentence(n_units, n_units * 3, bias=False)
        else:
            self.w_Q = ConvolutionSentence(n_units, n_units, bias=False)
            self.w_KV = ConvolutionSentence(n_units, n_units * 2, bias=False)
        self.finishing_linear_layer = ConvolutionSentence(n_units, n_units, bias=False)
        # attributes
        self.n_units = n_units
        self.n_heads = n_heads
        assert n_units % n_heads == 0
        self.scale_score = 1.0 / (n_units // n_heads) ** 0.5
        self.is_self_attention = self_attention
        if activation_fn == "softmax":
            self.activation = self._softmax_activation
        elif activation_fn == "relu":
            self.activation = self._relu_activation
        else:
            raise ValueError("unknown activation fn.")
        self.register_buffer("dummy_var", torch.empty(0))

    def forward(self, x, z=None, mask=None):
        n_units, n_heads = self.n_units, self.n_heads
        chunk_size = n_units // n_heads

        if self.is_self_attention:
            Q, K, V = torch.split(self.w_QKV(x), n_units, dim=1)
        else:
            Q = self.w_Q(x)
            K, V = torch.split(self.w_KV(z), n_units, dim=1)
        batch, n_units, n_queries = Q.shape
        _, _, n_keys = K.shape

        batch_Q = torch.cat(torch.split(Q, chunk_size, dim=1), dim=0)
        batch_K = torch.cat(torch.split(K, chunk_size, dim=1), dim=0)
        batch_V = torch.cat(torch.split(V, chunk_size, dim=1), dim=0)
        # assert batch_Q.shape == (batch * n_heads, n_units // n_heads, n_queries)
        # assert batch_K.shape == (batch * n_heads, n_units // n_heads, n_keys)
        # assert batch_V.shape == (batch * n_heads, n_units // n_heads, n_keys)

        batch_A = torch.bmm(batch_Q.permute(0, 2, 1), batch_K)
        batch_A = self.activation(batch_A, mask, batch, n_queries, n_keys)
        # assert batch_A.shape == (batch * n_heads, n_queries, n_keys)

        batch_A, batch_V = torch.broadcast_tensors(batch_A[:, None], batch_V[:, :, None])
        batch_C = torch.sum(batch_A * batch_V, axis=3)
        # assert batch_C.shape == (batch * n_heads, chunk_size, n_queries)
        C = torch.cat(torch.split(batch_C, batch, dim=0), dim=1)
        # assert C.shape == (batch, n_units, n_queries)
        C = self.finishing_linear_layer(C)
        return C

    def _softmax_activation(self, _batch_A, _mask, batch, n_queries, n_keys):
        mask = torch.cat([_mask] * self.n_heads, dim=0)

        batch_A = _batch_A * self.scale_score
        batch_A = torch.where(
            mask, batch_A, torch.full(batch_A.shape, float("-inf"), dtype=torch.float32, device=self.dummy_var.device)
        )
        batch_A = torch.softmax(batch_A, dim=2)
        batch_A = torch.where(torch.isnan(batch_A), torch.zeros_like(batch_A), batch_A)
        return batch_A

    def _relu_activation(self, _batch_A, _mask, batch, n_queries, n_keys):
        m = _mask.sum(dim=2).repeat_interleave(n_keys).reshape(batch, n_queries, n_keys)
        n_elements = torch.where(_mask, m, torch.ones_like(m))
        n_elements = torch.cat([n_elements] * self.n_heads, dim=0)
        mask = torch.cat([_mask] * self.n_heads, dim=0)

        batch_A = torch.relu(_batch_A)
        batch_A *= self.scale_score
        batch_A = torch.where(
            mask, batch_A, torch.full(batch_A.shape, float("-inf"), dtype=torch.float32, device=self.dummy_var.device)
        )
        batch_A /= n_elements
        batch_A = torch.where(torch.isinf(batch_A.data), torch.zeros_like(batch_A), batch_A)
        return batch_A

    def get_attnmap(self, x, z=None, mask=None):
        n_units, n_heads = self.n_units, self.n_heads
        chunk_size = n_units // n_heads

        if self.is_self_attention:
            Q, K, _ = torch.split(self.w_QKV(x), n_units, dim=1)
        else:
            Q = self.w_Q(x)
            K, _ = torch.split(self.w_KV(z), n_units, dim=1)
        batch, n_units, n_queries = Q.shape
        _, _, n_keys = K.shape

        batch_Q = torch.cat(torch.split(Q, chunk_size, dim=1), dim=0)
        batch_K = torch.cat(torch.split(K, chunk_size, dim=1), dim=0)

        batch_A = torch.bmm(batch_Q.permute(0, 2, 1), batch_K)
        return self.activation(batch_A, mask, batch, n_queries, n_keys)


class MultiHeadSimilarity(nn.Module):
    def __init__(self, n_units, n_heads=8):
        super(MultiHeadSimilarity, self).__init__()
        self.w_Q = ConvolutionSentence(n_units, n_units, bias=False)
        self.w_K = ConvolutionSentence(n_units, n_units, bias=False)
        self.ln = LayerNormalizationSentence(n_units, eps=1e-6)
        if n_heads > 1:
            self.finishing_linear_layer = ConvolutionSentence(n_units, n_units, bias=False)
        else:
            self.finishing_linear_layer = nn.Identity()
        self.n_units = n_units
        self.n_heads = n_heads
        assert n_units % n_heads == 0
        self.scale_score = 1.0 / (n_units // n_heads) ** 0.5

    def forward(self, x, y, xy_mask):
        """
        This function calculates
          x_i = LN(0.5*(q(x_i) + (1/n_y) Σ_j ReLU(q(x_i)^T k(y_j))k(y_j))), where j=[1, ..., n_y].
        The matrix representation is
         X = LN(X + (1/y_counts)ReLU(QK^T)V)
        mask: (batch, n_x, n_z)
        """
        n_units, n_heads = self.n_units, self.n_heads
        chunk_size = n_units // n_heads

        Q = self.w_Q(x)
        K = self.w_K(y)
        batch, n_units, _ = Q.shape

        n_elements = xy_mask.sum(dim=2)
        n_elements = torch.where(n_elements == 0, torch.ones_like(n_elements), n_elements)

        batch_Q = torch.cat(torch.split(Q, chunk_size, dim=1), dim=0)
        batch_K = torch.cat(torch.split(K, chunk_size, dim=1), dim=0)
        # assert batch_Q.shape == (batch * n_heads, chunk_size, n_queries)
        # assert batch_K.shape == (batch * n_heads, chunk_size, n_keys)

        mask = torch.cat([xy_mask] * n_heads, dim=0)
        batch_A = torch.bmm(batch_Q.permute(0, 2, 1), batch_K)
        batch_A = torch.relu(batch_A)
        batch_A *= self.scale_score
        batch_A = torch.where(mask, batch_A, torch.zeros_like(batch_A))
        # assert batch_A.shape == (batch * n_heads, n_queries, n_keys)
        batch_C = torch.bmm(batch_A, batch_K.permute(0, 2, 1))
        # assert batch_C.shape == (batch * n_heads, n_queries, chunk_size)
        C = torch.cat(torch.split(batch_C, batch, dim=0), dim=2)
        C = C.permute(0, 2, 1)
        # assert C.shape == (batch, n_units, n_queries)
        C /= n_elements[:, None, :]

        E = 0.5 * (Q + C)
        E = self.finishing_linear_layer(E)
        return E

    def get_attnmap(self, x, z, mask):
        n_units, n_heads = self.n_units, self.n_heads
        chunk_size = n_units // n_heads

        Q = self.w_Q(x)
        K = self.w_K(z)

        batch_Q = torch.cat(torch.split(Q, chunk_size, dim=1), dim=0)
        batch_K = torch.cat(torch.split(K, chunk_size, dim=1), dim=0)

        mask = torch.cat([mask] * n_heads, dim=0)
        batch_A = torch.bmm(batch_Q.permute(0, 2, 1), batch_K)
        batch_A = torch.relu(batch_A)
        batch_A *= self.scale_score
        batch_A = torch.where(mask, batch_A, torch.zeros_like(batch_A))
        return batch_A


class MultiHeadExpectation(nn.Module):
    def __init__(self, n_units, n_heads=8):
        super(MultiHeadExpectation, self).__init__()
        self.w = ConvolutionSentence(n_units, n_units, bias=False)
        self.fc = nn.Linear(n_heads, 1)
        self.n_units = n_units
        self.n_heads = n_heads
        assert n_units % n_heads == 0
        self.scale_score = 1.0 / (n_units // n_heads) ** 0.5

    def forward(self, x, y, xy_mask):
        # yx_mask: (batch, n_x, n_y)
        n_units, n_heads = self.n_units, self.n_heads
        chunk_size = n_units // n_heads

        batch, n_units, n_elem_x = x.shape
        _, _, n_elem_y = y.shape

        _x = self.w(x)
        _y = self.w(y)

        n_elements = torch.sum(xy_mask, dim=(1, 2))
        n_elements = torch.where(n_elements == 0, torch.ones_like(n_elements), n_elements)

        batch_x = torch.cat(torch.split(_x, chunk_size, dim=1), dim=0)
        batch_y = torch.cat(torch.split(_y, chunk_size, dim=1), dim=0)
        # assert batch_x.shape == (batch * n_heads, chunk_size, n_elem_x)
        # assert batch_y.shape == (batch * n_heads, chunk_size, n_elem_y)

        mask = torch.cat([xy_mask] * n_heads, dim=0)
        batch_xy = torch.bmm(batch_x.permute(0, 2, 1), batch_y)
        batch_xy = torch.relu(batch_xy)
        batch_xy *= self.scale_score
        batch_xy = torch.where(mask, batch_xy, torch.zeros_like(batch_xy))
        xy = torch.cat(torch.split(batch_xy[:, :, :, None], batch, dim=0), dim=3)
        similarity = torch.squeeze(torch.sum(xy, dim=(1, 2)))
        expectation = similarity / n_elements[:, None]
        expectation = self.fc(expectation)
        return expectation


class SAB(nn.Module):
    def __init__(self, n_units, n_heads=8, apply_ln=True):
        super(SAB, self).__init__()
        self.self_attention = MultiHeadAttention(n_units, n_heads, activation_fn="softmax")
        self.feed_forward = FeedForwardLayer(n_units)
        if apply_ln:
            self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)
        else:
            self.ln_1 = nn.Identity()
            self.ln_2 = nn.Identity()

    def forward(self, e, xx_mask):
        sub = self.self_attention(e, mask=xx_mask)
        e = e + sub
        e = self.ln_1(e)
        sub = self.feed_forward(e)
        e = e + sub
        e = self.ln_2(e)
        return e

    def get_attnmap(self, x, xx_mask):
        attn = self.self_attention.get_attnmap(x, mask=xx_mask)
        return attn


class ISAB(nn.Module):
    def __init__(self, n_units, n_heads=8, m=16):
        super(ISAB, self).__init__()
        self.paramI = nn.Parameter(torch.rand(n_units, m))
        self.self_attention_1 = MultiHeadAttention(n_units, n_heads, self_attention=False)
        self.feed_forward_1 = FeedForwardLayer(n_units)
        self.ln_1_1 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.ln_1_2 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.self_attention_2 = MultiHeadAttention(n_units, n_heads, self_attention=False)
        self.feed_forward_2 = FeedForwardLayer(n_units)
        self.ln_2_1 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.ln_2_2 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.m = m

    def forward(self, x, ix_mask, xi_mask):
        """
        ISAB(X) = MAB(X, H) \in R^{n \times d}
        where H = MAB(I, X) \ in R^{m \times d}
        MAB(u, v) = LayerNorm(H + rFF(H))
        where H = LayerNorm(u + Multihead(u, v, v))
        """
        batch, n_units, _ = x.shape
        # MAB(I, X), mask: m -> n
        i = torch.broadcast_to(self.paramI, (batch, n_units, self.m))
        mha_ix = self.self_attention_1(i, x, mask=ix_mask)
        h = self.ln_1_1(i + mha_ix)
        rff = self.feed_forward_1(h)
        h = self.ln_1_2(h + rff)
        # MAB(X, H), mask: n -> m
        mha_xh = self.self_attention_2(x, h, mask=xi_mask)
        h = self.ln_2_1(x + mha_xh)
        rff = self.feed_forward_2(h)
        h = self.ln_2_2(h + rff)
        return h


class MAB(nn.Module):
    """
    MAB(X, Y) = LN(H + rFF(H))
    where H = LN(X + Attention(X, Y, Y))
    """

    def __init__(self, n_units, n_heads=8, activation_fn="relu", apply_ln=True):
        super(MAB, self).__init__()
        self.attention = MultiHeadAttention(n_units, n_heads, self_attention=False, activation_fn=activation_fn)
        self.rff = FeedForwardLayer(n_units)
        if apply_ln:
            self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)
        else:
            self.ln_1 = nn.Identity()
            self.ln_2 = nn.Identity()

    def forward(self, x, y, mask):
        h = self.ln_1(x + self.attention(x, y, mask))
        e = self.ln_2(h + self.rff(h))
        return e


class PMA(nn.Module):
    """
    PMAk(Z) = MAB(S, rFF(Z))
            = LN(H + rFF(H))
    where H = LN(S + Multihead(S, rFF(Z), rFF(Z)))
    """

    def __init__(self, n_units, n_heads=8, n_output_instances=1, mh_activation="softmax"):
        self.k = n_output_instances
        super(PMA, self).__init__()
        if n_output_instances is None:
            self.S = None
        else:
            self.S = nn.Parameter(torch.rand(n_units, n_output_instances))
        self.attention = MultiHeadAttention(n_units, n_heads, self_attention=False, activation_fn=mh_activation)
        self.rff_1 = FeedForwardLayer(n_units)
        self.rff_2 = FeedForwardLayer(n_units)
        self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)

    def forward(self, z, mask, s=None):
        batch, n_units, _ = z.shape
        if s is None:
            s = torch.broadcast_to(self.S, (batch, n_units, self.k))
        else:
            k = s.shape[0]
            s = torch.broadcast_to(s.permute(1, 0), (batch, n_units, k))

        e = self.rff_1(z)
        h = self.ln_1(s + self.attention(s, e, mask))
        e = self.ln_2(h + self.rff_2(h))
        return e


class SetEncoder(nn.Module):
    def __init__(self, n_units, n_layers=2, n_heads=8, apply_ln=True):
        super(SetEncoder, self).__init__()
        self.layers = nn.ModuleList([SAB(n_units, n_heads, apply_ln=apply_ln) for _ in range(n_layers)])

    def forward(self, e, mask):
        for layer in self.layers:
            e = layer(e, mask)
        return e

    def get_attnmap(self, e, mask):
        attnmaps = []
        for layer in self.layers:
            attnmaps.append(layer.get_attnmap(e, mask))
            e = layer(e, mask)
        return attnmaps


class SetISABEncoder(nn.Module):
    def __init__(self, n_units, n_layers=2, n_heads=8, m=16):
        super(SetISABEncoder, self).__init__()
        self.layers = nn.ModuleList([ISAB(n_units, n_heads, m=m) for _ in range(n_layers)])

    def forward(self, e, ix_mask, xi_mask):
        for layer in self.layers:
            e = layer(e, ix_mask, xi_mask)
        return e


class SetDecoder(nn.Module):
    """
    Decoder(Z) = rFF(SAB(PMAk(Z))) \in R^{k \times d}
    """

    def __init__(self, n_units, n_heads=8, n_output_instances=1, apply_pma=True, apply_sab=True):
        self.n_units = n_units
        super(SetDecoder, self).__init__()
        if apply_pma:
            self.pma = PMA(n_units, n_heads=n_heads, n_output_instances=n_output_instances)
        else:
            self.rff_1 = FeedForwardLayer(n_units)
            self.mha = MultiHeadAttention(n_units, n_heads, self_attention=False, activation_fn="softmax")
            self.pma = lambda x, mask: self.mha(
                torch.ones((x.shape[0], self.n_units, n_output_instances), dtype=torch.float32, device=x.device),
                self.rff_1(x),
                mask,
            )
        if apply_sab:
            self.sab = SAB(n_units, n_heads)
        else:
            self.sab = lambda x, mask: x
        self.rff = FeedForwardLayer(n_units)

    def forward(self, z, yx_mask, yy_mask):
        e = self.pma(z, yx_mask)
        e = self.sab(e, yy_mask)
        e = self.rff(e)
        return e


class CrossSetDecoder(nn.Module):
    def __init__(
        self,
        n_units,
        n_heads=8,
        apply_last_rff=True,
        component="MHSim",
        activation_fn="relu",
        apply_ln=True,
    ):
        super(CrossSetDecoder, self).__init__()
        self.rff_1 = FeedForwardLayer(n_units)
        if component == "MAB":
            self.mab = MAB(n_units, n_heads, activation_fn=activation_fn, apply_ln=apply_ln)
        elif component == "MHAtt":
            self.mab = MultiHeadAttention(n_units, n_heads, self_attention=False, activation_fn=activation_fn)
        elif component == "MHSim":
            self.mab = MultiHeadSimilarity(n_units, n_heads)
        else:
            raise ValueError("no definition for MAB.")
        if apply_last_rff:
            self.rff_2 = FeedForwardLayer(n_units)
        else:
            self.rff_2 = nn.Identity()

    def forward(self, x, y, xy_mask):
        # The size of s is (batch, u_units, batch), broad-casted or generated by MAB.
        # If broad-casted, the 3rd axis of s is just copied from old s: (batch, u_units).
        # In MAB generating case, s is composed of batch x batch feature vectors.
        s = self.rff_1(x)  # (batch, n_units*3, batch)
        s = self.mab(s, y, xy_mask)  # (batch, n_units, batch)
        s = self.rff_2(s)
        return s

    def get_attnmap(self, x, y, xy_mask):
        s = self.rff_1(x)  # (batch, n_units*3, batch)
        attnmap = self.mab.get_attnmap(s, y, xy_mask)  # (batch, n_units, batch)
        return attnmap


class StackedCrossSetDecoder(nn.Module):
    def __init__(self, n_units, n_layers=2, n_heads=8, component="MHSim", activation_fn="relu", apply_ln=True):
        super(StackedCrossSetDecoder, self).__init__()
        layers = []

        # middle layers
        for _ in range(1, n_layers):
            layer = CrossSetDecoder(
                n_units,
                n_heads,
                apply_last_rff=False,
                component=component,
                activation_fn=activation_fn,
                apply_ln=apply_ln,
            )
            layers.append(layer)

        # last layer
        layer = CrossSetDecoder(
            n_units,
            n_heads,
            apply_last_rff=True,
            component=component,
            activation_fn=activation_fn,
            apply_ln=apply_ln,
        )
        layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, x, y, xy_mask):
        if len(x.shape) == 2:
            batch, n_units, _ = y.shape
            n_candidates = x.shape[0]
            assert x.shape[1] == n_units
            x = torch.broadcast_to(x.permute(1, 0), (batch, n_units, n_candidates))
        s = x
        for layer in self.layers:
            s = layer(s, y, xy_mask)
        return s

    def get_attnmap(self, x, y, xy_mask):
        attnmap = []
        if len(x.shape) == 2:
            batch, n_units, _ = y.shape
            n_candidates = x.shape[0]
            assert x.shape[1] == n_units
            x = torch.broadcast_to(x.permute(1, 0), (batch, n_units, n_candidates))
        s = x
        for layer in self.layers:
            attnmap.append(layer.get_attnmap(s, y, xy_mask))
            s = layer(x, y, xy_mask)
        return attnmap
