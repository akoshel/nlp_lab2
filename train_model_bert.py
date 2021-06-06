import torch
from loguru import logger


def train(model, iterator, optimizer, criterion, clip, train_history=None, valid_history=None):
    model.train()

    epoch_loss = 0
    history = []
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        outputs = model(input_ids=src, decoder_input_ids=trg)
        loss, logits = outputs.loss, outputs.logits

        loss.backward()

        # Let's clip the gradient

        optimizer.step()

        epoch_loss += loss

        history.append(loss.cpu().data.numpy())
        # logger.info("Train loss {loss}", loss=history[-1])

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    history = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            outputs = model(input_ids=src, decoder_input_ids=trg)
            loss, logits = outputs.loss, outputs.logits

            epoch_loss += loss

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs