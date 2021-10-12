import torch
import torch.nn as nn

from set_matching.models.cnn import CNN
from set_matching.models.modules import SetDecoder, SetEncoder, make_attn_mask


class SetTransformer(nn.Module):
    def __init__(
        self,
        n_units,
        n_encoder_layers=2,
        n_heads=8,
        n_output_instances=1,
        embedder_arch="resnet18",
        disable_cnn_update=False,
        dec_apply_pma=True,
        dec_apply_sab=True,
    ):
        super(SetTransformer, self).__init__()
        if embedder_arch == "linear":
            self.embedder = nn.Linear(4096, n_units)
        else:
            self.embedder = CNN(n_units, embedder_arch, disable_cnn_update)
        self.encoder = SetEncoder(n_units, n_layers=n_encoder_layers, n_heads=n_heads)
        self.decoder = SetDecoder(
            n_units,
            n_heads=n_heads,
            n_output_instances=n_output_instances,
            apply_pma=dec_apply_pma,
            apply_sab=dec_apply_sab,
        )
        self.n_units = n_units
        self.n_output_instances = n_output_instances
        self.register_buffer("_y_mask", torch.ones((1, self.n_output_instances), dtype=torch.bool))

    def forward(self, x, x_mask, t):
        batch, n_items = x.shape[:2]
        y_mask = torch.cat([self._y_mask] * batch, dim=0)
        xx_mask = make_attn_mask(x_mask, x_mask)
        yx_mask = make_attn_mask(y_mask, x_mask)
        yy_mask = make_attn_mask(y_mask, y_mask)

        x = self.embedder(x.reshape((-1,) + x.shape[2:]))  # (batch*n_items, n_units)
        x = x.reshape(batch, n_items, self.n_units).permute(0, 2, 1)  # (batch, n_units, n_items)

        z = self.encoder(x, xx_mask)
        # (batch, 512, 8)

        y = torch.squeeze(self.decoder(z, yx_mask, yy_mask), dim=2)
        # (batch, 512)

        t = self.embedder(t)
        # (batch, 512)

        score = torch.matmul(y, t.t())
        return score

    def predict(self, x, x_mask):
        batch, n_items = x.shape[:2]
        y_mask = torch.cat([self._y_mask] * batch, dim=0)
        xx_mask = make_attn_mask(x_mask, x_mask)
        yx_mask = make_attn_mask(y_mask, x_mask)
        yy_mask = make_attn_mask(y_mask, y_mask)

        x = self.embedder(x.reshape((-1,) + x.shape[2:]))  # (batch*n_items, n_units)
        x = x.reshape(batch, n_items, self.n_units).permute(0, 2, 1)  # (batch, n_units, n_items)

        z = self.encoder(x, xx_mask)
        # (batch, 512, 8)

        y = torch.squeeze(self.decoder(z, yx_mask, yy_mask), dim=2)
        # (batch, 512)

        return y

    def enc_dec(self, x, x_mask):
        batch = x.shape[0]
        y_mask = torch.cat([self._y_mask] * batch, dim=0)
        xx_mask = make_attn_mask(x_mask, x_mask)
        yx_mask = make_attn_mask(y_mask, x_mask)
        yy_mask = make_attn_mask(y_mask, y_mask)

        assert x.shape[2] == self.n_units
        x = x.permute(0, 2, 1)  # (batch, n_units, n_items)

        z = self.encoder(x, xx_mask)
        y = torch.squeeze(self.decoder(z, yx_mask, yy_mask), dim=2)

        return y

    def get_attnmap(self, x, x_mask):
        xx_mask = make_attn_mask(x_mask, x_mask)

        assert x.shape[2] == self.n_units
        x = x.permute(0, 2, 1)  # (batch, n_units, n_items)

        self.eval()
        attnmap = self.encoder.get_attnmap(x, xx_mask)

        return attnmap
