import torch
from set_matching.models.set_prediction import SetPrediction


def test_set_prediction():
    cardinality = 5
    n_units, n_encoder_layers, n_heads, n_iterations = 64, 2, 8, 2
    m = SetPrediction(
        cardinality, n_units, n_encoder_layers=n_encoder_layers, n_heads=n_heads, n_iterations=n_iterations
    )
    m.eval()

    batchsize, sentence_length, slot_length = 3, 8, 4
    x = torch.rand(batchsize, sentence_length, 3, 244, 244)
    x_mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4, [True] * 3 + [False] * 5])
    y_category = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 0], [4, 3, 0, 0]])
    y = torch.rand(batchsize, slot_length, 3, 244, 244)
    y = m(x, x_mask, y_category, y)
    assert y.shape == (batchsize * slot_length - 3, batchsize * slot_length - 3)
