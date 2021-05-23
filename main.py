import time
import math
import torch
import torch.nn as nn
from torch import optim
from torchtext.legacy.data import BucketIterator
import my_network
import network_gru_attention
import network_transformer
from load_data import get_dataset, split_data, _len_sort_key, save_vocab
from config import read_training_pipeline_params
from train_model import evaluate, train, epoch_time
import torchtext
from loguru import logger
import click
import numpy as np
from utils import generate_translation
import random
from helpers import get_bleu
from torch.utils.tensorboard import SummaryWriter




SEED = 2021

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def init_weights(m):
    # <YOUR CODE HERE>
    for name, param in m.named_parameters():
        nn.init.uniform_(param, -0.08, 0.08)


@click.command(name="main")
@click.argument("config_path")
def train_model(config_path: str):
    writer = SummaryWriter()
    config = read_training_pipeline_params(config_path)
    logger.info("pretrained_emb {b}", b=config.net_params.pretrained_emb)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device is {device}", device=device)
    SRC, TRG, dataset = get_dataset(config.dataset_path)
    train_data, valid_data, test_data = split_data(dataset, **config.split_ration.__dict__)
    if config.net_params.pretrained_emb:
        src_vectors = torchtext.vocab.FastText(language='ru')
    # trg_vectors = torchtext.vocab.FastText(language='en')
    # src_vectors = Vectors("cc.ru.300.bin", cache="cache")
    # trg_vectors = Vectors("wiki-news-300d-1M.vec", cache="cache")
    SRC.build_vocab(train_data, min_freq=3)
    if config.net_params.pretrained_emb:
        SRC.vocab.load_vectors(src_vectors)
    TRG.build_vocab(train_data, min_freq=3)
    torch.save(SRC.vocab, config.src_vocab_name)
    torch.save(TRG.vocab, config.trg_vocab_name)
    logger.info("Vocab saved")
    print(f"Unique tokens in source (ru) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=config.BATCH_SIZE,
        device=device,
        sort_key=_len_sort_key,
    )
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    if config.net_params.attention:
        Encoder = network_gru_attention.Encoder
        Decoder = network_gru_attention.Decoder
        Seq2Seq = network_gru_attention.Seq2Seq
        Attention = network_gru_attention.Attention
        attn = Attention(config.net_params.HID_DIM, config.net_params.HID_DIM)
        enc = Encoder(INPUT_DIM, config.net_params.ENC_EMB_DIM, config.net_params.HID_DIM, config.net_params.HID_DIM,
                      config.net_params.ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, config.net_params.DEC_EMB_DIM, config.net_params.HID_DIM, config.net_params.HID_DIM,
                      config.net_params.DEC_DROPOUT, attn)

        model = Seq2Seq(enc, dec, device)
    if config.net_params.transformer:
        logger.info("Transformer lets go")
        Encoder = network_transformer.Encoder
        Decoder = network_transformer.Decoder
        Seq2Seq = network_transformer.Seq2Seq
        train = network_transformer.train
        evaluate = network_transformer.evaluate
        SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
        TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
        HID_DIM = 256
        ENC_LAYERS = 3
        DEC_LAYERS = 3
        ENC_HEADS = 8
        DEC_HEADS = 8
        ENC_PF_DIM = 512
        DEC_PF_DIM = 512
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1

        enc = Encoder(INPUT_DIM,
                      HID_DIM,
                      ENC_LAYERS,
                      ENC_HEADS,
                      ENC_PF_DIM,
                      ENC_DROPOUT,
                      device)

        dec = Decoder(OUTPUT_DIM,
                      HID_DIM,
                      DEC_LAYERS,
                      DEC_HEADS,
                      DEC_PF_DIM,
                      DEC_DROPOUT,
                      device)
        model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    if False:
        Encoder = my_network.Encoder
        Decoder = my_network.Decoder
        Seq2Seq = my_network.Seq2Seq
        enc = Encoder(INPUT_DIM, config.net_params.ENC_EMB_DIM, config.net_params.HID_DIM,
                      config.net_params.N_LAYERS, config.net_params.ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, config.net_params.DEC_EMB_DIM, config.net_params.HID_DIM,
                          config.net_params.N_LAYERS, config.net_params.DEC_DROPOUT)
        model = Seq2Seq(enc, dec, device)

    model.apply(init_weights)
    if config.net_params.pretrained_emb:
        model.encoder.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(SRC.vocab.vectors))
    model.to(device)
    PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    optimizer = optim.Adam(model.parameters(), config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config.lr_scheduler.__dict__)
    train_history = []
    valid_history = []
    best_valid_loss = float('inf')
    print("Let's go")
    for p in model.encoder.parameters():
        p.requires_grad = True
    for p in model.decoder.parameters():
        p.requires_grad = True
    for epoch in range(config.N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, config.CLIP, train_history, valid_history)
        valid_loss = evaluate(model, valid_iterator, criterion)
        lr_scheduler.step(valid_loss)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), config.model_out_name)

        train_history.append(train_loss)
        valid_history.append(valid_loss)
        writer.add_scalar('train loss', train_history[-1], epoch)
        writer.add_scalar('valid loss', valid_history[-1], epoch)
        writer.add_scalar('learning rate', lr_scheduler._last_lr[0], epoch)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        for idx, batch in enumerate(valid_iterator):
            if idx > 3:
                break
            src = batch.src[:, idx:idx + 1]
            trg = batch.trg[:, idx:idx + 1]
            generate_translation(src, trg, model, TRG.vocab, SRC.vocab)

    get_bleu(model, test_iterator, TRG)


if __name__ == "__main__":
    train_model()
