import hydra
import torch
from hydra.utils import to_absolute_path
from tqdm import tqdm

from set_matching.datasets.popone_dataset import get_fitb_loader
from set_matching.datasets.split_dataset import get_fimbs_loader
from set_matching.models.set_transformer import SetTransformer
from set_matching.models.set_matching import SetMatching

MODELS = {"set_transformer": SetTransformer, "set_matching": SetMatching}
LOADERS = {"set_transformer": get_fitb_loader, "set_matching": get_fimbs_loader}


def predict_fitb(inputs, model):
    question, mask, answer = inputs
    batch, n_answers, _, _, insize = answer.shape

    y = model.predict(question, mask).unsqueeze(1)  # (batch, 1, n_units)
    t = model.embedder(answer.view(-1, 3, insize, insize))
    t = t.reshape(batch, n_answers, model.n_units).permute(0, 2, 1)

    score = torch.bmm(y, t).squeeze()
    pred = score.argmax(dim=1)

    return pred, torch.softmax(score, dim=1)


def predict_fimbs(inputs, model):
    query, q_mask, candidates, c_mask = inputs
    query_set_size = query.shape[1]
    batch, n_candidates, cand_set_size, _, _, insize = candidates.shape

    query = (
        torch.broadcast_to(query, (n_candidates,) + query.shape)
        .permute(1, 0, 2, 3, 4, 5)
        .reshape(-1, query_set_size, 3, insize, insize)
    )
    q_mask = torch.broadcast_to(q_mask, (n_candidates,) + q_mask.shape).permute(1, 0, 2).reshape(-1, query_set_size)
    candidates = candidates.view(-1, cand_set_size, 3, insize, insize)
    c_mask = c_mask.view(-1, cand_set_size)

    score = model(query, q_mask, candidates, c_mask)  # (batch*n_cands, batch*n_cands)
    score = torch.diagonal(score, 0).view(batch, n_candidates)
    pred = score.argmax(dim=1)

    return pred, torch.softmax(score, dim=1)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_config = dict(cfg.model)
    del model_config["name"]
    model = MODELS[cfg.model.name](**model_config)
    model.load_state_dict(torch.load(cfg.eval.modelckpt))
    model.to(device)
    model.eval()

    dataset_config = {
        "data_dir": to_absolute_path(cfg.dataset.data_dir),
        "batch_size": cfg.dataset.batch_size,
        "cnn_arch": model_config["cnn_arch"],
    }
    if cfg.model.name == "set_transformer":
        dataset_config["fname"] = "iqon_test_fitb.json"
        dataset_config["max_set_size"] = cfg.dataset.max_set_size
    elif cfg.model.name == "set_matching":
        dataset_config["fname"] = "iqon_test_finbs.json"
        dataset_config["max_set_size_query"] = cfg.dataset.max_set_size
        dataset_config["max_set_size_answer"] = cfg.eval.max_cand_set_size
    loader = LOADERS[cfg.model.name](**dataset_config)

    predict_fn = {"set_transformer": predict_fitb, "set_matching": predict_fimbs}[cfg.model.name]
    correct = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            pred, _ = predict_fn(tuple(map(lambda x: x.to(device), batch)), model)
            correct += pred.eq(torch.zeros_like(pred)).sum().item()

    print(f"Accuracy: {100 * correct / len(loader.dataset)}")


if __name__ == "__main__":
    main()
