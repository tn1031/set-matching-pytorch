def stopping_score_function(engine):
    val_loss = engine.state.metrics["loss"]
    return -val_loss


def lr_step(engine, lr_scheduler):
    lr_scheduler.step()


def log_training_loss(engine, loader, writer):
    print(
        "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
        "".format(
            engine.state.epoch,
            engine.state.iteration,
            len(loader),
            engine.state.output,
        )
    )
    writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)


def log_training_results(engine, name, evaluator, loader, history, writer):
    evaluator.run(loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics["acc"]
    avg_loss = metrics["loss"]
    print(
        "{} Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
            name, engine.state.epoch, avg_accuracy, avg_loss
        )
    )
    history["loss"].append(avg_loss)
    history["acc"].append(avg_accuracy)
    writer.add_scalar("{}/avg_accuracy".format(name), avg_accuracy, engine.state.epoch)
    writer.add_scalar("{}/avg_loss".format(name), avg_loss, engine.state.epoch)
