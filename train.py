import hydra
import torch
from hydra.utils import to_absolute_path
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss, RunningAverage
from tensorboardX import SummaryWriter

import set_matching.extensions as exfn
from set_matching.datasets.iqon3000_dataset import get_loader
from set_matching.models.set_matching import SetMatching
from set_matching.models.set_transformer import SetTransformer

MODELS = {"set_transformer": SetTransformer, "set_matching": SetMatching}


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # model
    model_config = dict(cfg.model)
    del model_config["name"]
    model = MODELS[cfg.model.name](**model_config)
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.5, weight_decay=0.00004)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.2)

    # dataset
    dataset_config = {
        "data_dir": to_absolute_path(cfg.dataset.data_dir),
        "batch_size": cfg.dataset.batch_size,
        "embedder_arch": model_config["embedder_arch"],
        "max_set_size": cfg.dataset.max_set_size,
    }
    if cfg.model.name == "set_matching":
        dataset_config["n_mix"] = cfg.dataset.n_mix
    dataset_name = "IQON3000"
    train_loader = get_loader(
        task_name=cfg.model.name,
        fname=f"{dataset_name}_train.json",
        **dataset_config,
        is_train=True,
    )
    val_loader = get_loader(
        task_name=cfg.model.name,
        fname=f"{dataset_name}_valid.json",
        **dataset_config,
        is_train=False,
    )

    # logger
    writer = SummaryWriter(logdir=cfg.logdir)

    def train_process(engine, batch):
        model.train()
        optimizer.zero_grad()
        batch = tuple(map(lambda x: x.to(device), batch))
        score = model(*batch)
        loss = loss_fn(score, torch.arange(score.size()[0]).to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        return loss.item()

    def eval_process(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(map(lambda x: x.to(device), batch))
            score = model(*batch)
            return score, torch.arange(score.size()[0]).to(device)

    trainer = Engine(train_process)
    train_evaluator = Engine(eval_process)
    valid_evaluator = Engine(eval_process)
    train_history = {"loss": [-1], "acc": [-1]}  # to avoid Index out of range.
    valid_history = {"loss": [-1], "acc": [-1]}

    # metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    Accuracy().attach(train_evaluator, "acc")
    Loss(loss_fn).attach(train_evaluator, "loss")
    Accuracy().attach(valid_evaluator, "acc")
    Loss(loss_fn).attach(valid_evaluator, "loss")

    # early stopping
    handler = EarlyStopping(
        patience=5,
        score_function=exfn.stopping_score_function,
        trainer=trainer,
    )
    valid_evaluator.add_event_handler(Events.COMPLETED, handler)

    # lr scheduler
    trainer.add_event_handler(Events.EPOCH_COMPLETED, exfn.lr_step, lr_scheduler)

    # logging
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=100),
        exfn.log_training_loss,
        train_loader,
        writer,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        exfn.log_training_results,
        "Training",
        train_evaluator,
        train_loader,
        train_history,
        writer,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        exfn.log_training_results,
        "Validation",
        valid_evaluator,
        val_loader,
        valid_history,
        writer,
    )

    # checkpoints
    objects_to_checkpoint = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    trainer_checkpointer = Checkpoint(
        to_save=objects_to_checkpoint,
        save_handler=DiskSaver(cfg.logdir, require_empty=False),
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=cfg.checkpoint_interval),
        trainer_checkpointer,
    )

    model_checkpointer = ModelCheckpoint(
        cfg.logdir,
        f"modelckpt_{cfg.model.name}",
        n_saved=1,
        create_dir=True,
        require_empty=False,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=cfg.checkpoint_interval),
        model_checkpointer,
        {"model": model},
    )

    if cfg.resume is not None:
        checkpoint = torch.load(cfg.resume)
        Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)

    # kick everything off
    trainer.run(train_loader, max_epochs=cfg.train.epochs)

    writer.close()


if __name__ == "__main__":
    main()
