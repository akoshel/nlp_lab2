import time
import math
import torch
import torch.nn as nn
from torch import optim
from torchtext.legacy.data import BucketIterator
import my_network
from load_data import get_dataset, split_data, _len_sort_key
from config import read_training_pipeline_params
from train_model import evaluate, train, epoch_time
import torchtext
from loguru import logger
from torchtext.vocab import Vectors
import network_gru_attention
import numpy as np
from utils import generate_translation
np.random.seed(2021)


def init_weights(m):
    # <YOUR CODE HERE>
    for name, param in m.named_parameters():
        nn.init.uniform_(param, -0.08, 0.08)


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device is {device}", device=device)
    SRC, TRG, dataset = get_dataset(config.dataset_path)
    train_data, valid_data, test_data = split_data(dataset, **config.split_ration.__dict__)
    src_vectors = torchtext.vocab.FastText(language='ru')
    # trg_vectors = torchtext.vocab.FastText(language='en')
    # src_vectors = Vectors("cc.ru.300.bin", cache="cache")
    # trg_vectors = Vectors("wiki-news-300d-1M.vec", cache="cache")
    SRC.build_vocab(train_data, min_freq=3)
    SRC.vocab.load_vectors(src_vectors)
    TRG.build_vocab(train_data, min_freq=3)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=config.BATCH_SIZE,
        device=device,
        sort_key=_len_sort_key
    )

    Encoder = my_network.Encoder
    Decoder = my_network.Decoder
    Seq2Seq = my_network.Seq2Seq
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    # enc = Encoder(INPUT_DIM, config.net_params.ENC_EMB_DIM, config.net_params.HID_DIM,
    #               config.net_params.N_LAYERS, config.net_params.ENC_DROPOUT)
    # dec = Decoder(OUTPUT_DIM, config.net_params.DEC_EMB_DIM, config.net_params.HID_DIM,
    #                config.net_params.N_LAYERS, config.net_params.DEC_DROPOUT)
    # enc.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(SRC.vocab.vectors))
    # dont forget to put the model to the right device
    # model = Seq2Seq(enc, dec, device).to(device)
    Encoder = network_gru_attention.EncoderRNN
    Decoder = network_gru_attention.AttnDecoderRNN
    Seq2Seq = network_gru_attention.Seq2SeqAttn
    enc = Encoder(INPUT_DIM, config.net_params.HID_DIM, device)
    dec = Decoder(config.net_params.HID_DIM, OUTPUT_DIM, config.net_params.DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    PAD_IDX = TRG.vocab.stoi['<pad>']
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

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        for idx, batch in enumerate(valid_iterator):
            if idx > 3:
                break
            src = batch.src[:, idx:idx + 1]
            trg = batch.trg[:, idx:idx + 1]
            generate_translation(src, trg, model, TRG.vocab, SRC.vocab)



if __name__ == "__main__":
    config = read_training_pipeline_params("train_config.yaml")
    train_model(config)
