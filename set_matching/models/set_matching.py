import torch
import torch.nn as nn

from set_matching.models.cnn import CNN
from set_matching.models.modules import MultiHeadExpectation, SetEncoder, StackedCrossSetDecoder, make_attn_mask


class SetMatching(nn.Module):
    def __init__(
        self,
        n_units,
        n_encoder_layer=1,
        n_decoder_layer=1,
        n_heads=8,
        n_iterative=2,
        enc_apply_ln=True,
        dec_apply_ln=True,
        dec_component="MHSim",
        embedder_arch="resnet18",
        disable_cnn_update=False,
    ):
        super(SetMatching, self).__init__()
        if embedder_arch == "linear":
            self.embedder = nn.Linear(4096, n_units)
        else:
            self.embedder = CNN(n_units, embedder_arch, disable_cnn_update)
        self.layers = []
        for i in range(0, n_iterative):
            name = f"enc_{i}"
            layer = SetEncoder(n_units, n_layers=n_encoder_layer, n_heads=n_heads, apply_ln=enc_apply_ln)
            setattr(self, name, layer)
            name = f"dec_{i}"
            layer = StackedCrossSetDecoder(
                n_units,
                n_layers=n_decoder_layer,
                n_heads=n_heads,
                component=dec_component,
                activation_fn="relu",
                apply_ln=dec_apply_ln,
            )
            setattr(self, name, layer)
        self.last_dec = MultiHeadExpectation(n_units, n_heads=n_heads)

        self.n_units = n_units
        self.n_iterative = n_iterative

    def forward(self, x, x_mask, y, y_mask):
        x = self.embed_reshape_transpose(x)
        y = self.embed_reshape_transpose(y)
        score = self.apply_enc_dec(x, x_mask, y, y_mask)
        return score

    def embed_reshape_transpose(self, x):
        batch, n_items = x.shape[:2]
        x = self.embedder(x.view((-1,) + x.shape[2:]))  # (batch*n_items, n_units)
        x = x.reshape(batch, n_items, self.n_units).permute(0, 2, 1)  # (batch, n_units, n_items)
        return x

    def apply_enc_dec(self, x, x_mask, y, y_mask):
        batch, n_units, n_x_items = x.shape
        n_y_items = y.shape[2]

        x = torch.reshape(
            torch.broadcast_to(torch.unsqueeze(x, dim=1), (batch, batch, n_units, n_x_items)),
            (batch * batch, n_units, n_x_items),
        )  # [x_1, x_1, ...]
        y = torch.reshape(
            torch.broadcast_to(torch.unsqueeze(y, dim=0), (batch, batch, n_units, n_y_items)),
            (batch * batch, n_units, n_y_items),
        )  # [y_1, y_2, ...]
        x_mask = torch.reshape(
            torch.broadcast_to(torch.unsqueeze(x_mask, dim=1), (batch, batch, n_x_items)), (batch * batch, n_x_items)
        )
        y_mask = torch.reshape(
            torch.broadcast_to(torch.unsqueeze(y_mask, axis=0), (batch, batch, n_y_items)), (batch * batch, n_y_items)
        )
        xx_mask = make_attn_mask(x_mask, x_mask)
        xy_mask = make_attn_mask(x_mask, y_mask)
        yy_mask = make_attn_mask(y_mask, y_mask)
        yx_mask = make_attn_mask(y_mask, x_mask)

        for i in range(0, self.n_iterative):
            x = getattr(self, f"enc_{i}")(x, xx_mask)  # (batch**2, n_units, n_x_items)
            y = getattr(self, f"enc_{i}")(y, yy_mask)  # (batch**2, n_units, n_drops)
            x_ = x + getattr(self, f"dec_{i}")(x, y, xy_mask)  # (batch**2, n_units, n_x_items)
            y_ = y + getattr(self, f"dec_{i}")(y, x, yx_mask)  # (batch**2, n_units, n_drops)
            x = x_
            y = y_

        e = self.last_dec(x, y, xy_mask)
        e = e.reshape(batch, batch)
        return e

    def get_attnmap(self, x, x_mask, y, y_mask):
        batch = x.shape[0]
        n_x_items = x.shape[1]
        n_y_items = y.shape[1]
        n_units = self.n_units

        x = self.embed_reshape_transpose(x)
        y = self.embed_reshape_transpose(y)

        # Extract set features by the encoder
        x = torch.reshape(
            torch.broadcast_to(torch.unsqueeze(x, dim=1), (batch, batch, n_units, n_x_items)),
            (batch * batch, n_units, n_x_items),
        )  # [x_1, x_1, ...]
        y = torch.reshape(
            torch.broadcast_to(torch.unsqueeze(y, dim=0), (batch, batch, n_units, n_y_items)),
            (batch * batch, n_units, n_y_items),
        )  # [y_1, y_2, ...]
        x_mask = torch.reshape(
            torch.broadcast_to(torch.unsqueeze(x_mask, dim=1), (batch, batch, n_x_items)), (batch * batch, n_x_items)
        )
        y_mask = torch.reshape(
            torch.broadcast_to(torch.unsqueeze(y_mask, dim=0), (batch, batch, n_y_items)), (batch * batch, n_y_items)
        )
        xx_mask = make_attn_mask(x_mask, x_mask)
        xy_mask = make_attn_mask(x_mask, y_mask)
        yy_mask = make_attn_mask(y_mask, y_mask)
        yx_mask = make_attn_mask(y_mask, x_mask)

        attnmap_x2y, attnmap_y2x = [], []
        for i in range(0, self.n_iterative):
            x = getattr(self, f"enc_{i}")(x, xx_mask)  # (batch**2, n_units, n_x_items)
            y = getattr(self, f"enc_{i}")(y, yy_mask)  # (batch**2, n_units, n_drops)
            x2y = getattr(self, f"dec_{i}").get_attnmap(x, y, xy_mask)
            y2x = getattr(self, f"dec_{i}").get_attnmap(y, x, yx_mask)
            x_ = x + getattr(self, f"dec_{i}")(x, y, xy_mask)  # (batch**2, n_units, n_x_items)
            y_ = y + getattr(self, f"dec_{i}")(y, x, yx_mask)  # (batch**2, n_units, n_drops)
            x = x_
            y = y_
            attnmap_x2y.append(x2y)
            attnmap_y2x.append(y2x)

        return attnmap_x2y, attnmap_y2x
