import torch
import torch.nn as nn

from set_matching.models.cnn import CNN
from set_matching.models.modules import SetEncoder, SlotAttention, make_attn_mask

PADDING_IDX = 0


class SetPrediction(nn.Module):
    def __init__(
        self,
        cardinality,
        n_units,
        n_encoder_layers=2,
        n_heads=8,
        n_iterations=3,
        embedder_arch="resnet18",
        disable_cnn_update=False,
    ):
        super(SetPrediction, self).__init__()
        if embedder_arch == "linear":
            self.embedder = nn.Linear(4096, n_units)
        else:
            self.embedder = CNN(n_units, embedder_arch, disable_cnn_update)
        self.encoder = SetEncoder(n_units, n_layers=n_encoder_layers, n_heads=n_heads)
        self.slot_attention = SlotAttention(n_units, n_heads=n_heads, n_output_instances=1, n_iterations=n_iterations)
        self.lookup = nn.Embedding(cardinality, n_units, padding_idx=PADDING_IDX)
        self.n_units = n_units

    def forward(self, x, x_mask, y_category, y):
        batch, n_inputs = x.shape[:2]
        n_slots = y_category.shape[1]

        y_mask = y_category != PADDING_IDX
        xx_mask = make_attn_mask(x_mask, x_mask)
        yx_mask = make_attn_mask(y_mask, x_mask)

        x = self.embedder(x.reshape((-1,) + x.shape[2:]))  # (batch*n_items, n_units)
        x = x.reshape(batch, n_inputs, self.n_units).permute(0, 2, 1)  # (batch, n_units, n_items)

        z = self.encoder(x, xx_mask)
        # (batch, 512, 8)

        category_emb = self.lookup(y_category).permute(0, 2, 1)
        pred_y = (
            self.slot_attention(z, yx_mask, slots=category_emb)
            .permute(0, 2, 1)
            .reshape((batch * n_slots, self.n_units))
        )

        true_y = self.embedder(y.reshape((-1,) + y.shape[2:]))

        y_mask = y_mask.reshape((batch * n_slots,))
        pred_y = pred_y[y_mask]
        true_y = true_y[y_mask]
        score = torch.matmul(pred_y, true_y.t())
        return score

    def predict(self, x, x_mask, y_category):
        batch, n_items = x.shape[:2]

        y_mask = y_category != PADDING_IDX
        xx_mask = make_attn_mask(x_mask, x_mask)
        yx_mask = make_attn_mask(y_mask, x_mask)

        x = self.embedder(x.reshape((-1,) + x.shape[2:]))  # (batch*n_items, n_units)
        x = x.reshape(batch, n_items, self.n_units).permute(0, 2, 1)  # (batch, n_units, n_items)

        z = self.encoder(x, xx_mask)
        # (batch, 512, 8)

        category_emb = self.lookup(y_category).permute(0, 2, 1)
        pred_y = self.slot_attention(z, yx_mask, slots=category_emb)
        return pred_y

    def get_attnmap(self, x, x_mask):
        xx_mask = make_attn_mask(x_mask, x_mask)

        assert x.shape[2] == self.n_units
        x = x.permute(0, 2, 1)  # (batch, n_units, n_items)

        self.eval()
        attnmap = self.encoder.get_attnmap(x, xx_mask)

        return attnmap
